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

sys.path.append(".")
from data import IMAGENETNORMALIZE, prepare_additive_data
from tools.misc import gen_folder_name, set_seed
from cfg import *

# Function to count trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Function to compute pruning ratio
def get_pruning_ratio(model):
    total_weights = 0
    pruned_weights = 0
    for param in model.parameters():
        if param.requires_grad:
            mask = (param != 0).float()
            total_weights += param.numel()
            pruned_weights += (mask == 0).sum().item()
    pruning_ratio = pruned_weights / total_weights * 100
    return pruning_ratio

# Function to apply pruning mask
def apply_pruning_mask(param, k):
    if param.numel() == 0:
        print("Skipping empty parameter tensor.")
        return torch.ones_like(param)
    
    num_params_to_keep = int(param.numel() * k)
    threshold = torch.topk(torch.abs(param).flatten(), num_params_to_keep, largest=True)[0].min()
    mask = (torch.abs(param) >= threshold).float()
    param.data *= mask
    return mask

# Function to prune model (applies mask)
def prune_model(model, k):
    for param in model.parameters():
        if param.requires_grad:
            mask = apply_pruning_mask(param, k)
            param.data *= mask
            print(f"After pruning: Shape {param.shape}, Non-zero params: {(param.data != 0).sum().item()}")


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18", "resnet50", "vit_b16", "vit_l16", "convnext_tiny", "convnext_base", 
                                         "swin_base", "swin_tiny", "vgg16", "vgg19", "instagram"], required=True)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--dataset', choices=["covid", "cifar100", "flowers102"], required=True)
    p.add_argument('--data-path', type=str, help="Dictionary containing dataset paths")
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--mask_lr', type=float, default=1e-2)
    p.add_argument('--k', type=float, default=0.8)  # 剩余权重比例
    p.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0, cuda:1, cuda:2, cuda:3, cuda:4, cuda:5)')
    args = p.parse_args()

    device = "cuda:5" if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    exp = f"cnn/linear_probing_bilevel"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Lambda(lambda x: x.convert('RGB') if hasattr(x, 'convert') else x),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])

    # print(f"Loading dataset: {args.dataset}")
    # data_path_dict = eval(args.data_path)
    # data_path = data_path_dict[args.dataset]
    # loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path, preprocess=preprocess)
    print(f"Loading dataset: {args.dataset}")
    loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path[args.dataset], preprocess=preprocess)

    # Load Network
    # if args.network == "resnet18":
    #     network = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1).to(device)
    #     in_features = network.fc.in_features
    #     network.fc = torch.nn.Linear(in_features, len(class_names))
    # elif args.network == "resnet50":
    #     network = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(device)
    #     in_features = network.fc.in_features
    #     network.fc = torch.nn.Linear(in_features, len(class_names))
    # elif args.network == "convnext_tiny":
    #     network = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1).to(device)
    #     network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, len(class_names)).to(device)
    # elif args.network == "convnext_base":
    #     network = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1).to(device)
    #     network.classifier[2] = torch.nn.Linear(network.classifier[2].in_features, len(class_names)).to(device)
    # elif args.network == "vit_b16":
    #     network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
    #     in_features = network.heads.head.in_features
    #     network.heads.head = torch.nn.Linear(in_features, len(class_names))
    # elif args.network == "vit_l16":
    #     network = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device)
    #     in_features = network.heads.head.in_features
    #     network.heads.head = torch.nn.Linear(in_features, len(class_names))
    # elif args.network == "vgg16":
    #     network = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).to(device)
    #     in_features = network.classifier[6].in_features
    #     network.classifier[6] = torch.nn.Linear(in_features, len(class_names)).to(device)
    # elif args.network == "vgg19":
    #     network = vgg19(weights=VGG19_Weights.IMAGENET1K_V1).to(device)
    #     in_features = network.classifier[6].in_features
    #     network.classifier[6] = torch.nn.Linear(in_features, len(class_names)).to(device)
    # elif args.network == "swin_base":
    #     network = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1).to(device)
    #     in_features = network.head.in_features
    #     network.head = torch.nn.Linear(in_features, len(class_names)).to(device)
    # elif args.network == "swin_tiny":
    #     network = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(device)
    #     in_features = network.head.in_features
    #     network.head = torch.nn.Linear(in_features, len(class_names)).to(device)
    # else:
    #     raise NotImplementedError(f"{args.network} is not supported")

    # network.requires_grad_(False)
    # if args.network in ["resnet18", "resnet50", "instagram"]:
    #     params = network.fc.parameters()
    # elif args.network in ["vit_b16", "vit_l16"]:
    #     params = network.heads.head.parameters()
    # elif args.network in ["convnext_tiny", "convnext_base"]:
    #     params = network.classifier[2].parameters()
    # elif args.network in ["vgg16", "vgg19"]:
    #     params = network.classifier[6].parameters()
    # elif args.network in ["swin_base", "swin_tiny"]:
    #     params = network.head.parameters()

    # for param in params:
    #     param.requires_grad = True

    # optimizer = torch.optim.Adam(params, lr=args.lr)
    # mask_optimizer = torch.optim.SGD(params, lr=args.mask_lr)
    # scaler = GradScaler()

    if args.network == "vit_b16":
        network = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1).to(device)
        in_features = network.heads.head.in_features
        network.heads.head = torch.nn.Linear(in_features, len(class_names)).to(device)
        network.requires_grad_(False)
        network.heads.head.requires_grad_(True)
        params = list(network.heads.head.parameters())  # 获取所有线性层的参数
    elif args.network == "vit_l16":
        network = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1).to(device)
        network.heads.head = torch.nn.Linear(network.heads.head.in_features, len(class_names)).to(device)
        network.requires_grad_(False)
        network.heads.head.requires_grad_(True)
        params = list(network.heads.head.parameters())

    elif args.network == "swin_tiny":
        network = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1).to(device)
        network.head = torch.nn.Linear(network.head.in_features, len(class_names)).to(device)
        network.requires_grad_(False)
        network.head.requires_grad_(True)
        params = list(network.head.parameters())

    elif args.network == "swin_base":
        network = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1).to(device)
        network.head = torch.nn.Linear(network.head.in_features, len(class_names)).to(device)
        network.requires_grad_(False)
        network.head.requires_grad_(True)
        params = list(network.head.parameters())

    else:
        raise NotImplementedError(f"{args.network} is not supported")

    print(f"Params to be optimized ({args.network}): {[p.shape for p in params if p.requires_grad]}")
    print(f"Total trainable parameters: {count_trainable_parameters(network)}")

    def count_nonzero_trainable_parameters(model):
        total_nonzero_params = sum((p.data != 0).sum().item() for p in model.parameters() if p.requires_grad)
        return total_nonzero_params


    nonzero_trainable_params = count_nonzero_trainable_parameters(network)
    print(f"非零可训练参数量: {nonzero_trainable_params}")


# 打印调试信息
    # print(f"Params to be optimized (vit_b16): {[p.shape for p in params if p.requires_grad]}")
    # if not params:
    #     raise ValueError("No parameters to optimize! Check if the head layer is correctly set.")

# 优化器
    optimizer = torch.optim.Adam(params, lr=args.lr)
    mask_optimizer = torch.optim.SGD(params, lr=args.mask_lr)
    scaler = GradScaler()


    # Training Loop
    for epoch in range(args.epoch):
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch} Training", ncols=100)

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mask_optimizer.zero_grad()

            # 下层优化
            with autocast():
                logits = network(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 上层优化
            logits = network(x)
            loss_mask = F.cross_entropy(logits, y)
            loss_mask.backward()
            mask_optimizer.step()

        # 剪枝操作
        prune_model(network, args.k)
        pruning_ratio = get_pruning_ratio(network)
        print(f"Epoch {epoch}: Pruning Ratio = {pruning_ratio:.2f}%")

        # 验证集
        network.eval()
        total_num, true_num = 0, 0
        with torch.no_grad():
            for x, y in loaders['test']:
                x, y = x.to(device), y.to(device)
                logits = network(x)
                total_num += y.size(0)
                true_num += torch.argmax(logits, dim=1).eq(y).sum().item()
        val_acc = 100 * true_num / total_num
        print(f"Epoch {epoch}, Validation Accuracy: {val_acc:.2f}%")
