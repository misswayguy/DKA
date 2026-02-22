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

from peft_model.adapter import ViTWithAdapter_2, ViTWithAdapterAndAlpha  # 模型导入

def count_trainable_parameters(model):
    """
    统计模型中可训练参数的数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=["vit_b16_adapter", "vit_l16_adapter", "vit_b16_adapter_alpha", "vit_l16_adapter_alpha"], required=True)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dataset', choices=["covid_0","covid_1","covid_5","covid_10","covid_20","covid_80",
                                         "covid_full", "covid"], required=True)
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training')
    args = parser.parse_args()

    # 设置设备和随机种子
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 保存路径设置
    exp = "vit_adapter_tuning"  # 实验名称
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    print(f"Loading dataset: {args.dataset}")
    loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path[args.dataset], preprocess=preprocess)

    # 加载模型
    num_classes = len(class_names)
    if args.network == "vit_b16_adapter":
        network = ViTWithAdapter_2(backbone="vit_b16", pretrained=True, freeze_backbone=True,
                                 num_classes=num_classes).to(device)
    elif args.network == "vit_l16_adapter":
        network = ViTWithAdapter_2(backbone="vit_l16", pretrained=True, freeze_backbone=True,
                                 num_classes=num_classes, selected_layers=list(range(0, 24))).to(device)
    elif args.network == "vit_b16_adapter_alpha":
        network = ViTWithAdapterAndAlpha(backbone="vit_b16", pretrained=True, freeze_backbone=True,
                                         num_classes=num_classes, initial_alpha=1.0).to(device)
    elif args.network == "vit_l16_adapter_alpha":
        network = ViTWithAdapterAndAlpha(backbone="vit_l16", pretrained=True, freeze_backbone=True,
                                         num_classes=num_classes, selected_layers=list(range(0, 24)), initial_alpha=1.0).to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    trainable_params = count_trainable_parameters(network)
    print(f"Total trainable parameters: {trainable_params}")

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # 混合精度训练
    scaler = GradScaler()
    best_acc = 0.0

    # 训练和验证循环
    for epoch in range(args.epoch):
        # 训练阶段
        network.train()
        total_loss, total_correct, total_samples = 0, 0, 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Epoch {epoch+1}/{args.epoch}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = network(x)
                loss = F.cross_entropy(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_loss += loss.item() * y.size(0)
            total_correct += (outputs.argmax(dim=1) == y).sum().item()
            total_samples += y.size(0)
            pbar.set_postfix_str(f"Loss: {total_loss / total_samples:.4f}, Acc: {100 * total_correct / total_samples:.2f}%")

        scheduler.step()
        logger.add_scalar("train/loss", total_loss / total_samples, epoch)
        logger.add_scalar("train/acc", total_correct / total_samples, epoch)

        # 验证阶段
        network.eval()
        y_true, y_pred, y_prob = [], [], []
        total_correct, total_samples = 0, 0
        with torch.no_grad():
            pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Testing", ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                preds = outputs.argmax(dim=1)
                probs = F.softmax(outputs, dim=1)[:, 1] if num_classes == 2 else None

                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                if probs is not None:
                    y_prob.extend(probs.cpu().numpy())

                total_correct += (preds == y).sum().item()
                total_samples += y.size(0)

        acc = total_correct / total_samples
        f1 = f1_score(y_true, y_pred, average='weighted')
        auc = roc_auc_score(y_true, y_prob) if num_classes == 2 else None
        logger.add_scalar("test/acc", acc, epoch)
        logger.add_scalar("test/f1", f1, epoch)
        if auc is not None:
            logger.add_scalar("test/auc", auc, epoch)

        print(f"Epoch {epoch+1}: Acc = {100 * acc:.2f}%, F1 = {f1:.4f}, AUC = {auc:.4f}" if auc else f"Epoch {epoch+1}: Acc = {100 * acc:.2f}%, F1 = {f1:.4f}")

        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(network.state_dict(), os.path.join(save_path, 'best.pth'))

        torch.save(network.state_dict(), os.path.join(save_path, 'last_epoch.pth'))

    print(f"Training complete. Best accuracy: {100 * best_acc:.2f}%")
