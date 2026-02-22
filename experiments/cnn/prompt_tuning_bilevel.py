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
from peft_model.prompt.vit_prompt import ViTWithPrompt  # 导入 ViT Prompt 模型

# Function to count trainable parameters
def count_trainable_parameters(params):
    return sum(p.numel() for p in params if p.requires_grad)

# Function to count non-zero trainable parameters
def count_nonzero_trainable_parameters(params):
    return sum((p.data != 0).sum().item() for p in params if p.requires_grad)

def count_total_and_nonzero_params(model):
    total_params = 0
    non_zero_params = 0
    for param in model.parameters():
        if param.requires_grad:
            total_params += param.numel()
            non_zero_params += (param != 0).sum().item()
    return total_params, non_zero_params

# Function to compute pruning ratio
def get_pruning_ratio(params):
    total_weights = 0
    pruned_weights = 0
    for param in params:
        if param.requires_grad:
            mask = (param != 0).float()
            total_weights += param.numel()
            pruned_weights += (mask == 0).sum().item()
    pruning_ratio = pruned_weights / total_weights * 100
    return pruning_ratio

# Function to apply pruning mask
def apply_pruning_mask(param, k):
    if param.numel() == 0:
        return torch.ones_like(param)
    num_params_to_keep = int(param.numel() * k)
    threshold = torch.topk(torch.abs(param).flatten(), num_params_to_keep, largest=True)[0].min()
    mask = (torch.abs(param) >= threshold).float()
    param.data *= mask
    return mask

# Function to prune model (applies mask)
def prune_model(params, k):
    for param in params:
        if param.requires_grad:
            mask = apply_pruning_mask(param, k)
            param.data *= mask

def get_prompt_params(network):
    """获取 Prompt 参数"""
    return [network.prompt_tokens]  # 将 prompt_tokens 封装成列表，便于传入优化器

if __name__ == '__main__':
    # 解析参数
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["vit_b16_prompt", "vit_l16_prompt"], required=True)
    p.add_argument('--dataset', choices=["cifar10", "cifar100", "covid"], required=True)
    p.add_argument('--epoch', type=int, default=200)
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--mask_lr', type=float, default=1e-2)
    p.add_argument('--k', type=float, default=0.8)  # 剩余权重比例
    p.add_argument('--virtual_tokens', type=int, default=5, help="Number of virtual tokens for Prompt")
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', type=str, default="cuda:0")
    args = p.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 生成实验路径
    exp = "cnn/prompt_tuning_bilevel"
    save_path = os.path.join(results_path, exp, gen_folder_name(args))
    os.makedirs(save_path, exist_ok=True)
    logger = SummaryWriter(os.path.join(save_path, 'tensorboard'))

    # 数据预处理
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
    ])
    print(f"Loading dataset: {args.dataset}")
    loaders, class_names = prepare_additive_data(args.dataset, data_path=data_path[args.dataset], preprocess=preprocess)
    num_classes = len(class_names)

    # 加载网络
    if args.network == "vit_b16_prompt":
        network = ViTWithPrompt(backbone="vit_b16", num_prompts=args.virtual_tokens, num_classes=num_classes, freeze_backbone=True).to(device)
    elif args.network == "vit_l16_prompt":
        network = ViTWithPrompt(backbone="vit_l16", num_prompts=args.virtual_tokens, num_classes=num_classes, freeze_backbone=True).to(device)
    else:
        raise NotImplementedError(f"Network {args.network} not supported.")

    # 获取 Prompt Token 参数
    prompt_params = get_prompt_params(network)

    # print(f"Trainable parameters (total): {count_trainable_parameters(prompt_params)}")
    # print(f"Non-zero trainable parameters: {count_nonzero_trainable_parameters(prompt_params)}")

    print(f"Trainable parameters (total): {count_trainable_parameters(network.parameters())}")
    print(f"Non-zero trainable parameters: {count_nonzero_trainable_parameters(network.parameters())}")

    # 优化器
    optimizer = torch.optim.Adam(prompt_params, lr=args.lr)
    mask_optimizer = torch.optim.SGD(prompt_params, lr=args.mask_lr)
    scaler = GradScaler()

    # Training Loop
    for epoch in range(args.epoch):
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch} Training", ncols=100)

        # print(f"Epoch {epoch} - Trainable parameters: {count_trainable_parameters(prompt_params)}")
        # print(f"Epoch {epoch} - Non-zero trainable parameters: {count_nonzero_trainable_parameters(prompt_params)}")
        total_params, non_zero_params = count_total_and_nonzero_params(network)
        print(f"Epoch {epoch}: Total trainable parameters: {total_params}")
        print(f"Epoch {epoch}: Non-zero trainable parameters: {non_zero_params}")

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

            # 上层优化 (针对 prompt tokens)
            logits = network(x)
            loss_mask = F.cross_entropy(logits, y)
            loss_mask.backward()
            mask_optimizer.step()

            total_num += y.size(0)
            true_num += torch.argmax(logits, dim=1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)
            pbar.set_postfix_str(f"Acc: {100 * true_num / total_num:.2f}%, Loss: {loss_sum / total_num:.4f}")

        logger.add_scalar("train/accuracy", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)

        # 剪枝操作
        prune_model(prompt_params, args.k)
        pruning_ratio = get_pruning_ratio(prompt_params)
        print(f"Epoch {epoch}: Pruning Ratio = {pruning_ratio:.2f}%")

        # 验证阶段
        network.eval()
        total_num, true_num = 0, 0
        with torch.no_grad():
            pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epoch {epoch} Testing", ncols=100)
            for x, y in pbar:
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                total_num += y.size(0)
                true_num += outputs.argmax(dim=1).eq(y).sum().item()
                pbar.set_postfix_str(f"Acc: {100 * true_num / total_num:.2f}%")

        acc = true_num / total_num
        logger.add_scalar("test/accuracy", acc, epoch)

        print(f"Epoch {epoch} - Validation: Trainable parameters: {count_trainable_parameters(prompt_params)}")
        print(f"Epoch {epoch} - Validation: Non-zero trainable parameters: {count_nonzero_trainable_parameters(prompt_params)}")

        # 保存模型
        if acc > best_acc:
            best_acc = acc
            torch.save(network.state_dict(), os.path.join(save_path, "best.pth"))
        torch.save(network.state_dict(), os.path.join(save_path, "last.pth"))

    print(f"Training complete. Best accuracy: {100 * acc:.2f}%")
