import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import transforms
from torchvision.models import (
    resnet18, ResNet18_Weights, resnet50, ResNet50_Weights,
    vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights,
    vgg16, VGG16_Weights, vgg19, VGG19_Weights, swin_b, swin_t, Swin_B_Weights, Swin_T_Weights
)
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import sys
from sklearn.metrics import f1_score, roc_auc_score


sys.path.append(".")
from data import IMAGENETNORMALIZE, prepare_additive_data
from tools.misc import gen_folder_name, set_seed
from cfg import *

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18", "resnet50", "vit_b16", "vit_l16", "convnext_tiny", "convnext_base", 
                                         "swin_base", "swin_tiny", "vgg16", "vgg19", "instagram"], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=[ "covid_0","covid_1","covid_5","covid_10","covid_20","covid_80",
                                         "covid_full", "covid"], required=True)
    p.add_argument('--epoch', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.1)
    p.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0, cuda:1, cuda:2, cuda:3, cuda:4, cuda:5)')
    args = p.parse_args()

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    exp = f"cnn/linear_probing"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])

    print(f"Loading dataset: {args.dataset}")
    loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path[args.dataset], preprocess=preprocess)

    # Load Network
    if args.network == "resnet18":
        network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
        in_features = network.fc.in_features
        network.fc = torch.nn.Linear(in_features, len(class_names))
    elif args.network == "resnet50":
        network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
        in_features = network.fc.in_features
        network.fc = torch.nn.Linear(in_features, len(class_names))
    elif args.network == "convnext_tiny":
        network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
        network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, len(class_names)).to(device)
    elif args.network == "convnext_base":
        network = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1).to(device)
        network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, len(class_names)).to(device)
    elif args.network == "vit_b16":
        network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        in_features = network.heads.head.in_features
        network.heads.head = torch.nn.Linear(in_features, len(class_names))
    elif args.network == "vit_l16":
        network = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device)
        in_features = network.heads.head.in_features
        network.heads.head = torch.nn.Linear(in_features, len(class_names))
    elif args.network == "vgg16":
        network = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
        in_features = network.classifier[6].in_features
        network.classifier[6] = torch.nn.Linear(in_features, len(class_names)).to(device)
    elif args.network == "vgg19":
        network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
        in_features = network.classifier[6].in_features
        network.classifier[6] = torch.nn.Linear(in_features, len(class_names)).to(device)
    elif args.network == "swin_base":
        network = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1).to(device)
        in_features = network.head.in_features
        network.head = torch.nn.Linear(in_features, len(class_names)).to(device)
    elif args.network == "swin_tiny":
        network = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(device)
        in_features = network.head.in_features
        network.head = torch.nn.Linear(in_features, len(class_names)).to(device)
    elif args.network == "instagram":
        from torch import hub
        network = hub.load('facebookresearch/WSL-Images', 'resnext101_32x8d_wsl').to(device)
        in_features = network.fc.in_features
        network.fc = torch.nn.Linear(in_features, len(class_names))
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    network.requires_grad_(False)  # Freeze backbone
    if args.network in ["resnet18", "resnet50", "instagram"]:
        network.fc.requires_grad_(True)
        optimizer = torch.optim.Adam(network.fc.parameters(), lr=args.lr)
    elif args.network in ["vit_b16", "vit_l16"]:
        network.heads.head.requires_grad_(True)
        optimizer = torch.optim.Adam(network.heads.head.parameters(), lr=args.lr)
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

    # 统计可训练参数
    trainable_params = count_trainable_parameters(network)
    print(f"Total trainable parameters: {trainable_params}")

    # Scheduler
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[int(0.5 * args.epoch), int(0.72 * args.epoch)], gamma=0.1)

    # Create Save Directory
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # Training Loop
    best_acc = 0.
    scaler = GradScaler()
    for epoch in range(args.epoch):
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

        # Validation
        # total_num = 0
        # true_num = 0
        # pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
        # for x, y in pbar:
        #     x, y = x.to(device), y.to(device)
        #     with torch.no_grad():
        #         fx = network(x)
        #     total_num += y.size(0)
        #     true_num += torch.argmax(fx, 1).eq(y).float().sum().item()
        #     acc = true_num / total_num
        #     pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")

        # logger.add_scalar("test/acc", acc, epoch)
        network.eval()
        y_true = []
        y_pred = []
        y_prob = []
        total_num = 0
        true_num = 0

        with torch.no_grad():
            pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epo {epoch} Testing", ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                fx = network(x)
                preds = torch.argmax(fx, dim=1)
                probs = F.softmax(fx, dim=1)[:, 1]  # 假设是二分类问题
                
                y_true.extend(y.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())

                total_num += y.size(0)
                true_num += preds.eq(y).float().sum().item()
                acc = true_num / total_num
                pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")

        f1 = f1_score(y_true, y_pred, average='weighted')  # 加权平均 F1

        logger.add_scalar("test/acc", acc, epoch)
        logger.add_scalar("test/f1", f1, epoch)
        # if auc is not None:
        #     logger.add_scalar("test/auc", auc, epoch)
        print(f"Epoch {epoch}: F1-score = {f1:.4f}")

        # Save Best Model
        state_dict = {
            "fc_dict": (
                network.fc.state_dict() if args.network in ["resnet18", "resnet50", "instagram"]
                else network.heads.head.state_dict() if args.network in ["vit_b16", "vit_l16"]
                else network.classifier.state_dict() if args.network in ["convnext_tiny", "convnext_base"]
                else None
            ),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        if acc > best_acc:
            best_acc = acc
            state_dict['best_acc'] = best_acc
            torch.save(state_dict, os.path.join(save_path, 'best.pth'))
        torch.save(state_dict, os.path.join(save_path, 'ckpt.pth'))

    print(f"Training complete. Best accuracy: {100 * best_acc:.2f}%")
