import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

import sys
sys.path.append(".")
from data import IMAGENETNORMALIZE, prepare_additive_data
from tools.misc import gen_folder_name, set_seed
from cfg import *

# 从register模块导入ViT模型
from peft_model.register.vit_register import ViTWithRegisterTokens

def count_trainable_parameters(model):
    """
    统计模型中可训练参数的数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["vit_b16_register", "vit_l16_register"], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=[ "covid"], required=True)
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--num_register_tokens', type=int, default=4, help="Number of register tokens to use.")
    p.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0, cuda:1, cuda:2, cuda:3, cuda:4, cuda:5)')
    args = p.parse_args()

    # 设置设备
    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 日志和保存路径
    exp = "cnn/vit_register_finetuning"
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

    # 加载 ViT + Register Tokens 网络
    num_classes = len(class_names)
    if args.network == "vit_b16_register":
        network = ViTWithRegisterTokens(
            backbone="vit_b16", pretrained=True, num_register_tokens=args.num_register_tokens,
            freeze_backbone=True, num_classes=num_classes
        ).to(device)
    elif args.network == "vit_l16_register":
        network = ViTWithRegisterTokens(
            backbone="vit_l16", pretrained=True, num_register_tokens=args.num_register_tokens,
            freeze_backbone=True, num_classes=num_classes
        ).to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    trainable_params = count_trainable_parameters(network)
    print(f"Total trainable parameters: {trainable_params}")

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), 
                                 lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # 混合精度训练
    scaler = GradScaler()
    best_acc = 0.0

    # 训练和验证循环
    for epoch in range(args.epoch):
        # 训练阶段
        network.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']), 
                    desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                fx = network(x)
                loss = F.cross_entropy(fx, y, reduction='mean')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            loss_sum += loss.item() * fx.size(0)
            pbar.set_postfix_str(f"Acc {100 * true_num / total_num:.2f}%")

        scheduler.step()
        logger.add_scalar("train/acc", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)

        # 测试阶段
        network.eval()
        total_num = 0
        true_num = 0
        with torch.no_grad():
            pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                fx = network(x)
                total_num += y.size(0)
                true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
                acc = true_num / total_num
                pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")

        logger.add_scalar("test/acc", acc, epoch)

        # 保存最佳模型
        state_dict = {
            "network_dict": network.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))

