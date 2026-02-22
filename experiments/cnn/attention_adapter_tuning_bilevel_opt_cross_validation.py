import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import sys
sys.path.append(".")
from peft_model.adapter import ResNetWithAdapter, ViTWithAdapter, SwinWithAdapter, ConvNeXtWithAdapter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略损坏的图片

from sklearn.metrics import f1_score, recall_score, classification_report

import time

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 数据加载函数
def prepare_custom_data(data_path, preprocess):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
    test_dataset = datasets.ImageFolder(root=test_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
    # test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    return {"train": train_loader, "test": test_loader}, class_names

# 计算可训练参数
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 获取 Adapter 和 Head 层的参数
def get_adapter_and_head_params(network):
    adapter_params = []
    head_params = []

    for name, param in network.named_parameters():
        if "adapter" in name:
            adapter_params.append(param)  # Adapter 层参数
        elif "head" in name or "classifier" in name:
            head_params.append(param)  # 分类层参数

    return adapter_params, head_params

def get_adapter_attention_head_params(network):
    adapter_params = []
    attention_params = []
    head_params = []

    for name, param in network.named_parameters():
        if "adapter" in name:
            adapter_params.append(param)
        elif "attn" in name or "attention" in name:
            attention_params.append(param)
        elif "head" in name or "classifier" in name:
            head_params.append(param)

    return adapter_params, attention_params, head_params


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=["resnet18_adapter", "vit_b16_adapter", "convnext_tiny_adapter",
                                              "vit_b16_adapter_low", "vit_b16_adapter_mid", "vit_b16_adapter_high","swin_base_adapter", "swin_tiny_adapter"], required=True)
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset")
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)  # Head 学习率（提升收敛速度）
    parser.add_argument('--mask_lr', type=float, default=1e-3)  # Adapter 学习率（降低以防梯度爆炸）
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda:0")
    args = parser.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    print(f"Loading dataset from {args.dataset}")
    loaders, class_names = prepare_custom_data(args.dataset, preprocess)
    num_classes = len(class_names)

    # 加载模型  
    model_dict = {  
        "resnet18_adapter": ResNetWithAdapter("resnet18", pretrained=True, reduction=64, freeze_backbone=True, num_classes=num_classes),
        "vit_b16_adapter": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=16, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(0, 12))),
        "vit_b16_adapter_low": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(0, 4))),
        "vit_b16_adapter_mid": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(4, 8))),
        "vit_b16_adapter_high": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(8, 12))),
        "swin_tiny_adapter": SwinWithAdapter("swin_t", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes),
        "swin_base_adapter": SwinWithAdapter("swin_b", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes),
        "convnext_tiny_adapter": ConvNeXtWithAdapter("convnext_tiny", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes),
    }
    network = model_dict[args.network].to(device)

    # 获取 Adapter 和 Head 层参数
    # adapter_params, head_params = get_adapter_and_head_params(network)
    
    adapter_params, attention_params, head_params = get_adapter_attention_head_params(network)
    
    adapter_and_attn_params = adapter_params + attention_params

    print(f"Total trainable parameters: {count_trainable_parameters(network)}")
    # 优化器
    optimizer = torch.optim.Adam(head_params, lr=args.lr)
    # mask_optimizer = torch.optim.Adam(adapter_params, lr=args.mask_lr)
    mask_optimizer = torch.optim.Adam(adapter_and_attn_params, lr=args.mask_lr)
    scaler = GradScaler()

    best_acc = 0.0

    # 训练循环
    for epoch in range(args.epoch):
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch} Training", ncols=100)

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mask_optimizer.zero_grad()

           
            with autocast():
                logits = network(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)  # 防止梯度爆炸

            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)  # Adapter 也裁剪梯度
            scaler.step(optimizer)
            scaler.step(mask_optimizer)
            scaler.update()

            # # Step 2: update adapter five times
            # if batch_idx % 5 == 0:
            #     logits = network(x)
            #     loss_mask = F.cross_entropy(logits, y)
            #     loss_mask.backward()
            #     torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)  # 防止梯度爆炸
            #     mask_optimizer.step()

            total_num += y.size(0)
            true_num += logits.argmax(1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)
            pbar.set_postfix_str(f"Acc: {100 * true_num / total_num:.2f}%, Loss: {loss_sum / total_num:.4f}")

        train_acc = true_num / total_num
        print(f"Epoch {epoch+1} - Train Loss: {loss_sum/total_num:.4f}, Train Acc: {train_acc*100:.2f}%")

        # 测试阶段
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
        
        total_num, true_num = 0, 0
        all_preds = []
        all_labels = []

        # with torch.no_grad():
        #     for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
        #         x, y = x.to(device), y.to(device)
        #         outputs = network(x)
        #         total_num += y.size(0)
        #         true_num += outputs.argmax(1).eq(y).sum().item()
        with torch.no_grad():
            for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total_num += y.size(0)
                true_num += preds.eq(y).sum().item()

        # test_acc = true_num / total_num
        # print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%")

        test_acc = true_num / total_num
        f1 = f1_score(all_labels, all_preds, average='macro')
        sensitivity = recall_score(all_labels, all_preds, average='macro')  # 多分类 Sensitivity

        print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}% | F1-score (macro): {f1:.4f} | Sensitivity (macro): {sensitivity:.4f}")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(network.state_dict(), "best_adapter_model.pth")

    print(f"Training complete. Best Accuracy: {best_acc*100:.2f}%")
