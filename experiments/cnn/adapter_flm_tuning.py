import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import numpy as np
from functools import partial

import sys
sys.path.append(".")
from data import IMAGENETNORMALIZE, prepare_additive_data
from algorithms import generate_label_mapping_by_frequency, label_mapping_base, get_dist_matrix
from tools.misc import gen_folder_name, set_seed
from tools.mapping_visualization import plot_mapping
from cfg import *
from peft_model.adapter.resnet_adapter_flm import ResNetWithAdapter_FLM  # 导入新修改的 ResNet + Adapter + FLM 模型

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18_adapter_flm", "resnet50_adapter_flm"], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102",
                                         "ucf101", "food101", "gtsrb", "svhn", "eurosat",
                                         "oxfordpets", "stanfordcars", "sun397", "covid"], required=True)
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=0.001)
    args = p.parse_args()

    # 设置设备和随机种子
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 日志和保存路径
    exp = "cnn/adapter_flm_tuning"
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

    # 加载 ResNet + Adapter + FLM 网络
    num_classes = len(class_names)
    if args.network == "resnet18_adapter_flm":
        network = ResNetWithAdapter_FLM(backbone="resnet18", pretrained=True, freeze_backbone=True,
                                        num_classes=num_classes, freeze_fc=True).to(device)
    elif args.network == "resnet50_adapter_flm":
        network = ResNetWithAdapter_FLM(backbone="resnet50", pretrained=True, freeze_backbone=True,
                                        num_classes=num_classes, freeze_fc=True).to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    # 优化器和学习率调度器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), 
                                 lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # 生成固定的映射序列
    print("Generating Frequency Label Mapping...")
    mapping_sequence = generate_label_mapping_by_frequency(
        visual_prompt=None,  # 如果 FLM 这里不使用 prompt，可以传 None
        network=network,
        data_loader=loaders['train'],  # 加载器替代 missing 参数
        mapping_num=1  # 默认为1，您可以自定义
    )
    label_mapping = partial(label_mapping_base, mapping_sequence=mapping_sequence)
    torch.save(mapping_sequence, os.path.join(save_path, 'mapping_sequence.pth'))

    # 混合精度训练
    scaler = GradScaler()
    best_acc = 0.0

    # 训练和测试循环
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
                fx = label_mapping(network(x))
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
                fx = label_mapping(network(x))
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
