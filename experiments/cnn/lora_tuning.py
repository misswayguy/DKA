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

# 导入 LoRA 模型
from peft_model.lora.resnet_lora import ResNetWithLoRA  # ResNet + LoRA
from peft_model.lora.vit_lora import ViTWithLoRA       # ViT + LoRA
from peft_model.lora.cnn_lora import BackboneWithLoRA
from peft_model.lora.swin_lora import SwinWithLoRA

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18_lora", "resnet50_lora", "convnext_tiny_lora", "convnext_base_lora", "swin_tiny_lora", "swin_base_lora",
                                          "vgg16_lora", "vgg19_lora", "vit_b16_lora", "vit_l16_lora"], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", 
                                         "ucf101", "food101", "gtsrb", "svhn", "eurosat", 
                                         "oxfordpets", "stanfordcars", "sun397", "covid"], required=True)
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--rank', type=int, default=1, help="LoRA rank")
    p.add_argument('--lora_alpha', type=float, default=1, help="LoRA alpha scaling factor")
    p.add_argument('--lora_dropout', type=float, default=0.99, help="LoRA dropout rate")
    args = p.parse_args()

    # 设备设置
    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    set_seed(args.seed)

    # 日志和保存路径
    exp = "cnn/lora_finetuning"  
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

    # 加载网络
    num_classes = len(class_names)
    if args.network == "resnet18_lora":
        network = BackboneWithLoRA(backbone="resnet18", pretrained=True, freeze_backbone=True, 
                                 num_classes=num_classes, rank=args.rank, 
                                 lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "resnet50_lora":
        network = BackboneWithLoRA(backbone="resnet50", pretrained=True, freeze_backbone=True, 
                                 num_classes=num_classes, rank=args.rank, 
                                 lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "convnext_tiny_lora":
        network = BackboneWithLoRA(backbone="convnext_tiny", pretrained=True, freeze_backbone=True, 
                                   num_classes=num_classes, rank=args.rank, 
                                   lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "convnext_base_lora":
        network = BackboneWithLoRA(backbone="convnext_base", pretrained=True, freeze_backbone=True, 
                                   num_classes=num_classes, rank=args.rank, 
                                   lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "vgg16_lora":
        network = BackboneWithLoRA(backbone="vgg16", pretrained=True, freeze_backbone=True, 
                                   num_classes=num_classes, rank=args.rank, 
                                   lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "vgg19_lora":
        network = BackboneWithLoRA(backbone="vgg19", pretrained=True, freeze_backbone=True, 
                                   num_classes=num_classes, rank=args.rank, 
                                   lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "vit_b16_lora":
        network = ViTWithLoRA(backbone="vit_b_16", pretrained=True, freeze_backbone=True, 
                              num_classes=num_classes, rank=args.rank, 
                              lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "vit_l16_lora":
        network = ViTWithLoRA(backbone="vit_l_16", pretrained=True, freeze_backbone=True, 
                              num_classes=num_classes, rank=args.rank, 
                              lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "swin_tiny_lora":
        network = SwinWithLoRA(backbone="swin_t", pretrained=True, freeze_backbone=True, 
                              num_classes=num_classes, rank=args.rank, 
                              lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    elif args.network == "swin_base_lora":
        network = SwinWithLoRA(backbone="swin_b", pretrained=True, freeze_backbone=True, 
                              num_classes=num_classes, rank=args.rank, 
                              lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout).to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    # 打印可训练参数量
    trainable_params = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print(f"Trainable parameters: {trainable_params}")

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
