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
from sklearn.metrics import f1_score, recall_score

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

    train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)  # 修改为 torchvision.datasets.ImageFolder
    test_dataset = datasets.ImageFolder(root=test_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    return {"train": train_loader, "test": test_loader}, class_names

# 修改后的数据加载函数 tiny imagenet
# def prepare_custom_data(data_path, preprocess):
#     train_path = os.path.join(data_path, "train")
#     val_path = os.path.join(data_path, "val")  # 注意这里改为val而不是test
    
#     train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
#     val_dataset = datasets.ImageFolder(root=val_path, transform=preprocess)  # 使用val作为测试集

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=64, shuffle=True, num_workers=4
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=64, shuffle=False, num_workers=4
#     )

#     class_names = train_dataset.classes
#     return {"train": train_loader, "test": val_loader}, class_names  # 返回val作为test

if __name__ == "__main__":
    # 参数解析
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=["vit_b16", "vit_l16", "convnext_tiny", "convnext_base", 
                                                "swin_base", "swin_tiny",], required=True)
    parser.add_argument('--dataset', type=str, required=True, help="数据集路径")
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0, cuda:1, cuda:2, cuda:3, cuda:4, cuda:5)')
    args = parser.parse_args()

    # 设备配置
    # device = "cuda:4" if torch.cuda.is_available() else "cpu"
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(42)

    # dataset_name = os.path.basename(os.path.normpath(args.dataset))
    dataset_name = os.path.basename(os.path.dirname(os.path.normpath(args.dataset)))



    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"Loading dataset from {args.dataset}")
    loaders, class_names = prepare_custom_data(args.dataset, preprocess)
    num_classes = len(class_names)

    # 计算可训练参数
    def count_trainable_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    # 加载模型
    if args.network == "vit_b16":
        network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        in_features = network.heads.head.in_features
        network.heads.head = torch.nn.Linear(in_features, num_classes).to(device)
    elif args.network == "vit_l16":
        network = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device)
        in_features = network.heads.head.in_features
        network.heads.head = torch.nn.Linear(in_features, num_classes).to(device)
    elif args.network == "convnext_tiny":
        network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
        network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, len(class_names)).to(device)
    elif args.network == "convnext_base":
        network = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1).to(device)
        network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, len(class_names)).to(device)
    elif args.network == "swin_base":
        network = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1).to(device)
        in_features = network.head.in_features
        network.head = torch.nn.Linear(in_features, len(class_names)).to(device)
    elif args.network == "swin_tiny":
        network = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(device)
        in_features = network.head.in_features
        network.head = torch.nn.Linear(in_features, len(class_names)).to(device)

    # # 冻结 backbone，只训练分类头
    # for param in network.parameters():
    #     param.requires_grad = False
    # for param in network.heads.head.parameters():
    #     param.requires_grad = True
    network.requires_grad_(False)  # Freeze backbone
    
    # 解冻所有 bias 参数
    for name, param in network.named_parameters():
        if "bias" in name:
            param.requires_grad = True
            
    if args.network in ["resnet18", "resnet50", "instagram"]:
        network.fc.requires_grad_(True)
        optimizer = torch.optim.Adam(network.fc.parameters(), lr=args.lr)
    elif args.network in ["vit_b16", "vit_l16"]:
        network.heads.head.requires_grad_(True)
        # optimizer = torch.optim.Adam(network.heads.head.parameters(), lr=args.lr)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, network.parameters()), lr=args.lr
        )

    elif args.network in ["convnext_tiny", "convnext_base"]:
        network.classifier[2].requires_grad_(True)
        optimizer = torch.optim.Adam(network.classifier[2].parameters(), lr=args.lr)
    elif args.network in ["vgg16", "vgg19"]:
        network.classifier[6].requires_grad_(True)
        optimizer = torch.optim.Adam(network.classifier[6].parameters(), lr=args.lr)
    elif args.network in ["swin_base", "swin_tiny"]:
        network.head.requires_grad_(True)
        optimizer = torch.optim.Adam(network.head.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    network = network.to(device)

    print(f"Total trainable parameters: {count_trainable_parameters(network)}")

    # optimizer = torch.optim.Adam(network.heads.head.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
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
        # total_correct, total_samples = 0, 0
        total_num, true_num = 0, 0

        all_preds = []
        all_labels = []

        # with torch.no_grad():
        #     for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
        #         x, y = x.to(device), y.to(device)
        #         outputs = network(x)
        #         total_correct += (outputs.argmax(1) == y).sum().item()
        #         total_samples += y.size(0)
        
        with torch.no_grad():
            for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total_num += y.size(0)
                true_num += preds.eq(y).sum().item()

        # test_acc = total_correct / total_samples
        test_acc = true_num / total_num
        f1 = f1_score(all_labels, all_preds, average='macro')
        sensitivity = recall_score(all_labels, all_preds, average='macro') 

        #print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%")
        print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%, F1: {f1:.4f}, Sensitivity: {sensitivity:.4f}")

        dataset_base_name = dataset_name.replace('/', '_')  # 用 _ 替代路径中的 / 以适应文件名

        # 构建保存路径
        save_dir = f'/mnt/data/lsy/ZZQ/{dataset_base_name}'

        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)


        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            # torch.save(network.state_dict(), "best_model.pth")
            torch.save(network.state_dict(), f"/mnt/data/lsy/ZZQ/best_model_{args.network}_{dataset_name}.pth")
            # torch.save(network.state_dict(), os.path.join(save_dir, f"best_model_{args.network}_{dataset_base_name}.pth"))

        scheduler.step()

    print(f"Training complete. Best Accuracy: {best_acc*100:.2f}%")
