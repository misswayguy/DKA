import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights,
    vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights,
    vgg16, VGG16_Weights, vgg19, VGG19_Weights, swin_b, swin_t, Swin_B_Weights, Swin_T_Weights
)

from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略损坏的图片

import time

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 加载数据集
def prepare_custom_data(data_path, preprocess):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
    test_dataset = datasets.ImageFolder(root=test_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    return {"train": train_loader, "test": test_loader}, class_names

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=["vit_b16", "vit_l16", "convnext_tiny", "convnext_base", 
                                            "swin_base", "swin_tiny"], required=True)
    parser.add_argument('--dataset', type=str, required=True, help="数据集路径")
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.001, help='学习率 (full finetuning通常比linear probing小)')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='权重衰减')
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training')
    args = parser.parse_args()

    # 设备配置
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(42)

    dataset_name = os.path.basename(os.path.dirname(os.path.normpath(args.dataset)))

    # 数据增强 (full finetuning通常需要更强的数据增强)
    preprocess = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    test_preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"Loading dataset from {args.dataset}")
    loaders, class_names = prepare_custom_data(args.dataset, preprocess)
    test_loaders, _ = prepare_custom_data(args.dataset, test_preprocess)  # 测试集使用不同的预处理
    num_classes = len(class_names)

    # 加载模型
    if args.network == "vit_b16":
        network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        in_features = network.heads.head.in_features
        network.heads.head = torch.nn.Linear(in_features, num_classes)
    elif args.network == "vit_l16":
        network = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1)
        in_features = network.heads.head.in_features
        network.heads.head = torch.nn.Linear(in_features, num_classes)
    elif args.network == "convnext_tiny":
        network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, num_classes)
    elif args.network == "convnext_base":
        network = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, num_classes)
    elif args.network == "swin_base":
        network = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1)
        in_features = network.head.in_features
        network.head = torch.nn.Linear(in_features, num_classes)
    elif args.network == "swin_tiny":
        network = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1)
        in_features = network.head.in_features
        network.head = torch.nn.Linear(in_features, num_classes)

    # Full finetuning: 解冻所有参数
    network.requires_grad_(True)
    network = network.to(device)

    # 优化器 (使用更小的学习率和权重衰减)
    optimizer = torch.optim.AdamW(network.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    
    # 学习率调度器 (使用余弦退火)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    scaler = GradScaler()

    # 训练
    best_acc = 0.0
    for epoch in range(args.epoch):
        network.train()
        total_loss, total_correct, total_samples = 0, 0, 0

        for x, y in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{args.epoch}", ncols=100):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = network(x)
                loss = F.cross_entropy(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * y.size(0)
            total_correct += (outputs.argmax(1) == y).sum().item()
            total_samples += y.size(0)

        train_acc = total_correct / total_samples
        print(f"Epoch {epoch+1} - Train Loss: {total_loss/total_samples:.4f}, Train Acc: {train_acc*100:.2f}%")

        # 验证
        network.eval()
        
        print("Profiling inference latency and memory usage...")
        
        test_iter = iter(loaders["test"])
        x, y = next(test_iter)
        x = x.to(device)

        # 预热，防止第一次测量慢
        for _ in range(10):
            _ = network(x)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()

        with torch.no_grad():
            _ = network(x)

        torch.cuda.synchronize()
        end_time = time.time()

        latency = (end_time - start_time) * 1000  # ms
        peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)  # MB

        print(f"Inference latency: {latency:.2f} ms | Peak memory: {peak_memory:.2f} MB")       
        
        total_correct, total_samples = 0, 0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for x, y in tqdm(test_loaders["test"], desc="Testing", ncols=100):
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                preds = outputs.argmax(1)
                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y.cpu().numpy())

        test_acc = total_correct / total_samples
        test_f1 = f1_score(all_targets, all_preds, average='macro')
        print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%, Test F1: {test_f1:.4f}")

        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(network.state_dict(), f"/mnt/data/lsy/ZZQ/best_model_ft_{args.network}_{dataset_name}.pth")
            print(f"New best model saved with accuracy {best_acc*100:.2f}%")

        scheduler.step()

    print(f"Training complete. Best Accuracy: {best_acc*100:.2f}%")