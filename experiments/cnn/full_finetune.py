import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights, vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights, vgg16, VGG16_Weights, vgg19, VGG19_Weights
from torchvision.models import  swin_b, swin_t, Swin_B_Weights, Swin_T_Weights

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

import sys
sys.path.append(".")
from data import IMAGENETNORMALIZE, prepare_additive_data
from tools.misc import gen_folder_name, set_seed
from cfg import *

def count_trainable_parameters(model):
    """
    统计模型中可训练参数的数量
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18", "resnet50", "convnext_tiny", "convnext_base", "vit_b16", "vit_l16", "instagram",
                                         "vgg16", "vgg19", "swin_base", "swin_tiny"], required=True)
    p.add_argument('--seed', type=int, default=7)
    #p.add_argument('--dataset', choices=["cifar10", "cifar100", "abide", "dtd", "flowers102", "ucf101", "food101", "gtsrb", "svhn", "eurosat", "oxfordpets", "stanfordcars", "sun397"], required=True)
    p.add_argument('--dataset', choices=["covid_full", "covid"], required=True)
    p.add_argument('--epoch', type=int, default=50)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0, cuda:1, cuda:2, cuda:3, cuda:4, cuda:5)')
    args = p.parse_args()

    device = "cuda:4" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    exp = f"cnn/full_finetuning"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    preprocess = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    #loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path, preprocess=preprocess)
    print(f"Loading dataset: {args.dataset}")
    loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path[args.dataset], preprocess=preprocess)
    # loaders, class_names = prepare_additive_data(
    #     dataset=args.dataset,
    #     data_path=data_path[args.dataset],
    #     preprocess=preprocess,
    #     #save_dir="/mnt/data/lsy/ZZQ/covid_limited_data",  # 明确指定保存路径
    #     train_ratio=0.01,
    #     test_ratio=0.2
    # )
    
    # Network
    if args.network == "resnet18":
        from torchvision.models import resnet18, ResNet18_Weights
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    elif args.network == "convnext_tiny":
        network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "convnext_base":
        network = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "vit_b16":
        network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "vit_l16":
        network = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "vgg16":
        network = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "vgg19":
        network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "swin_base":
        network = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "swin_tiny":
        network = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(device)
    elif args.network == "instagram":
        from torch import hub
        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")
    network.requires_grad_(True)
    network = network.to(device)

    trainable_params = count_trainable_parameters(network)
    print(f"Total trainable parameters: {trainable_params}")

    # Optimizer
    optimizer = torch.optim.Adam(network.parameters(), lr=args.lr, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # Make Dir
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Train
    best_acc = 0. 
    scaler = GradScaler()
    for epoch in range(args.epoch):
        network.train()
        total_num = 0
        true_num = 0
        loss_sum = 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']),
                desc=f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            pbar.set_description_str(f"Epo {epoch} Training Lr {optimizer.param_groups[0]['lr']:.1e}", refresh=True)
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
            pbar.set_postfix_str(f"Acc {100*true_num/total_num:.2f}%")
        scheduler.step()
        logger.add_scalar("train/acc", true_num/total_num, epoch)
        logger.add_scalar("train/loss", loss_sum/total_num, epoch)
        
        # Test
        network.eval()
        total_num = 0
        true_num = 0
        pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            with torch.no_grad():
                fx = network(x)
            total_num += y.size(0)
            true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
            acc = true_num/total_num
            pbar.set_postfix_str(f"Acc {100*acc:.2f}%")
        logger.add_scalar("test/acc", acc, epoch)

        # Save CKPT
        state_dict = {
            "network_dict": network.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))
