import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
from sklearn.metrics import f1_score, roc_auc_score

import sys
sys.path.append(".")
from data import IMAGENETNORMALIZE, prepare_additive_data
from tools.misc import gen_folder_name, set_seed
from cfg import *
from peft_model.adapter import ResNetWithAdapter, ViTWithAdapter, BackboneWithAdapter, SwinWithAdapter  # 导入 Adapter 模型

# Function to count trainable parameters
def count_trainable_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)

# Function to get adapter and head parameters separately
def get_adapter_and_head_params(network):
    adapter_params = []
    head_params = []

    for name, param in network.named_parameters():
        if "adapter" in name:
            adapter_params.append(param)  # Adapter 模块参数
        elif "head" in name or "classifier" in name:
            head_params.append(param)  # 分类头参数

    return adapter_params, head_params

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18_adapter", "resnet50_adapter", "convnext_tiny_adapter",
                                         "convnext_base_adapter", "swin_base_adapter", "swin_tiny_adapter",
                                         "vit_b16_adapter_11th","vit_b16_adapter_10_11th", "vit_b16_adapter_9_11th","vit_b16_adapter_8_11th",
                                          "vit_b16_adapter_7_11th","vit_b16_adapter_6_11th","vit_b16_adapter_5_11th","vit_b16_adapter_4_11th", "vit_b16_adapter_3_11th",
                                          "vit_b16_adapter_2_11th", "vit_b16_adapter_1_11th",
                                         "vit_b16_adapter", "vit_l16_adapter"], required=True)
    p.add_argument('--dataset', choices=["covid_0","covid_1","covid_5","covid_10","covid_20","covid_80","covid_full", "covid"], required=True)
    p.add_argument('--epoch', type=int, default=150)
    p.add_argument('--lr', type=float, default=1e-5)  # 下层学习率
    p.add_argument('--mask_lr', type=float, default=1e-3)  # 上层学习率 (Adapter)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', type=str, default="cuda:0")
    args = p.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 生成实验路径
    exp = "cnn/adapter_bilevel_training"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    print(f"Loading dataset: {args.dataset}")
    loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path[args.dataset], preprocess=preprocess)
    num_classes = len(class_names)

    # 加载 Adapter 网络
    if args.network == "resnet18_adapter":
        network = ResNetWithAdapter(backbone="resnet18", pretrained=True, reduction=64, freeze_backbone=True, num_classes=num_classes).to(device)
    elif args.network == "vit_b16_adapter":
        network = ViTWithAdapter(backbone="vit_b16", pretrained=True, 
                                 #reduction=768, 
                                 middle_dim=1,
                                 freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(0, 12))).to(device)
    elif args.network == "vit_b16_adapter_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=[11]  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_10_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(10, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_8_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(8, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_9_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(9, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_7_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(7, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_6_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(6, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_3_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(3, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_5_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(5, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_4_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(4, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_2_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(2, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "vit_b16_adapter_1_11th":
        network = ViTWithAdapter(
            backbone="vit_b16",
            pretrained=True,
            reduction=768,
            freeze_backbone=True,
            num_classes=num_classes,
            selected_layers=list(range(1, 12))  # 修改这里的层数范围
        ).to(device)
    elif args.network == "swin_tiny_adapter":
        network = SwinWithAdapter(backbone="swin_t", pretrained=True, reduction=96, freeze_backbone=True, num_classes=num_classes).to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    # 获取 Adapter 和分类头参数
    adapter_params, head_params = get_adapter_and_head_params(network)

    for name, param in network.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")

    # 打印可训练参数信息
    print(f"Total trainable parameters (Adapter + Head): {count_trainable_parameters(adapter_params + head_params)}")

    # 优化器
    optimizer = torch.optim.Adam(head_params, lr=args.lr)  # 下层优化器 (分类头参数)
    mask_optimizer = torch.optim.Adam(adapter_params, lr=args.mask_lr)  # 上层优化器 (Adapter 参数)
    scaler = GradScaler()

    # 训练循环
    best_acc = 0.0
    for epoch in range(args.epoch):
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch} Training", ncols=100)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mask_optimizer.zero_grad()

            # 下层优化 (更新分类头)
            with autocast():
                logits = network(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()

            # 检查梯度
            # for name, param in network.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(f"{name} grad: {param.grad}")

            scaler.step(optimizer)
            scaler.update()

            # 上层优化 (更新 Adapter)
            logits = network(x)
            loss_mask = F.cross_entropy(logits, y)
            loss_mask.backward()

            # 检查梯度
            # for name, param in network.named_parameters():
            #     if param.requires_grad and param.grad is not None:
            #         print(f"{name} grad: {param.grad}")
                    
            mask_optimizer.step()

            total_num += y.size(0)
            true_num += torch.argmax(logits, dim=1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)
            pbar.set_postfix_str(f"Acc: {100 * true_num / total_num:.2f}%, Loss: {loss_sum / total_num:.4f}")

        logger.add_scalar("train/accuracy", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)

        # 测试阶段
        network.eval()
        y_true, y_pred, y_prob = [], [], []
        total_num, true_num = 0, 0

        with torch.no_grad():
            pbar = tqdm(loaders['test'], desc=f"Epoch {epoch} Testing", ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                fx = network(x)
                preds = torch.argmax(fx, dim=1)
                probs = F.softmax(fx, dim=1)[:, 1]  # 二分类问题

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

                total_num += y.size(0)
                true_num += preds.eq(y).sum().item()
                acc = true_num / total_num
                pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")

        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_prob) if num_classes == 2 else None

        logger.add_scalar("test/accuracy", acc, epoch)
        logger.add_scalar("test/f1", f1, epoch)
        if auc is not None:
            logger.add_scalar("test/auc", auc, epoch)
        print(f"Epoch {epoch}: F1-score = {f1:.4f}, AUC = {auc:.4f}" if auc else f"Epoch {epoch}: F1-score = {f1:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(network.state_dict(), os.path.join(save_path, "best.pth"))
        torch.save(network.state_dict(), os.path.join(save_path, "last.pth"))

    print(f"Training complete. Best accuracy: {100 * best_acc:.2f}%")
