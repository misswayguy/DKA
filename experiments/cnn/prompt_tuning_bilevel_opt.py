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
from peft_model.prompt.vit_prompt import ViTWithPrompt  # 导入 ViT Prompt 模型
from peft_model.prompt.swin_prompt import SwinWithPrompt   # 导入 Swin Prompt 模型

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

# 获取 Prompt Tokens 和 Head 参数
# def get_prompt_and_head_params(network):
#     """获取 Prompt Tokens 和 Classifier Head 的参数"""
#     prompt_params = [network.prompt_tokens]
#     head_params = list(network.base_model.heads.head.parameters())
#     return prompt_params, head_params
def get_prompt_and_head_params(network):
    """获取 Prompt Tokens 和 Classifier Head 的参数"""
    prompt_params = []
    # Swin Transformer
    if isinstance(network, SwinWithPrompt):
        # 检查 PromptedSwinTransformer 是否存在 prompt_embeddings
        if hasattr(network.model, 'prompt_embeddings'):
            prompt_params.append(network.model.prompt_embeddings)
        # 还需要深层嵌入的 prompt tokens
        if hasattr(network.model, 'deep_prompt_embeddings_0'):
            prompt_params.extend([
                network.model.deep_prompt_embeddings_0,
                network.model.deep_prompt_embeddings_1,
                network.model.deep_prompt_embeddings_2,
                network.model.deep_prompt_embeddings_3
            ])
    # ViT Transformer
    elif isinstance(network, ViTWithPrompt):
        if hasattr(network, "prompt_tokens"):
            prompt_params = [network.prompt_tokens]

    # 分类头参数
    # head_params = list(network.model.head.parameters()) if hasattr(network.model, "head") else list(network.base_model.heads.head.parameters())
    head_params = list(network.base_model.heads.parameters())  # 修改为提取整个 heads
    
    return prompt_params, head_params


if __name__ == '__main__':
    # 解析参数
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["vit_b16_prompt", "vit_l16_prompt", 
                                         "vit_b16_prompt_11th","vit_b16_prompt_10th","vit_b16_prompt_9th","vit_b16_prompt_8th","vit_b16_prompt_7th",
                                         "vit_b16_prompt_6th","vit_b16_prompt_5th","vit_b16_prompt_4th","vit_b16_prompt_3th","vit_b16_prompt_2th","vit_b16_prompt_1th",
                                         "swin_tiny_prompt", "swin_base_prompt"], required=True)
    p.add_argument('--dataset', choices=["covid_full", "covid"], required=True)
    p.add_argument('--epoch', type=int, default=150)
    p.add_argument('--lr', type=float, default=1e-4)  # 下层学习率
    p.add_argument('--mask_lr', type=float, default=1e-2)  # 上层学习率
    p.add_argument('--k', type=float, default=1)  # 剩余权重比例
    p.add_argument('--virtual_tokens', type=int, default=1, help="Number of virtual tokens for Prompt")
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', type=str, default="cuda:0")
    args = p.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 生成实验路径
    exp = "/mnt/data/lsy/ZZQ/prompt_tuning_bilevel_opt"
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
        network = ViTWithPrompt(backbone="vit_b16", num_prompts=args.virtual_tokens, num_classes=num_classes, freeze_backbone=True,selected_blocks=[],).to(device)
    elif args.network == "vit_l16_prompt":
        network = ViTWithPrompt(backbone="vit_l16", num_prompts=args.virtual_tokens, num_classes=num_classes, freeze_backbone=True).to(device)
    elif args.network == "vit_b16_prompt_11th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[11],
        ).to(device)
    elif args.network == "vit_b16_prompt_10th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[10],
        ).to(device)
    elif args.network == "vit_b16_prompt_9th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[9],
        ).to(device)
    elif args.network == "vit_b16_prompt_8th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[8],
        ).to(device)
    elif args.network == "vit_b16_prompt_7th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[7],
        ).to(device)
    elif args.network == "vit_b16_prompt_6th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[6],
        ).to(device)
    elif args.network == "vit_b16_prompt_5th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[5],
        ).to(device)
    elif args.network == "vit_b16_prompt_4th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[4],
        ).to(device)
    elif args.network == "vit_b16_prompt_3th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[3],
        ).to(device)
    elif args.network == "vit_b16_prompt_2th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[2],
        ).to(device)
    elif args.network == "vit_b16_prompt_1th":
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[1],
        ).to(device)
    elif args.network == "swin_tiny_prompt":
        network = SwinWithPrompt(
            backbone="swin_t",
            num_virtual_tokens=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            pretrained_path="/home/lusiyuan/ZZQ/prompt/sw/swin_tiny_patch4_window7_224.pth",
        ).to(device)
    elif args.network == "swin_base_prompt":
        network = SwinWithPrompt(
            backbone="swin_b",
            num_virtual_tokens=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True,
            pretrained_path="/home/lusiyuan/ZZQ/prompt/sw/swin_base_patch4_window12_384_22k.pth", 
        ).to(device)
    else:
        raise NotImplementedError(f"Network {args.network} not supported.")

    # 获取 Prompt 和 Head 参数
    prompt_params, head_params = get_prompt_and_head_params(network)

    # 打印可训练参数信息
    print(f"Total trainable parameters (Prompt + Head): {count_trainable_parameters(prompt_params + head_params)}")
    print(f"Non-zero trainable parameters (Prompt + Head): {count_nonzero_trainable_parameters(prompt_params + head_params)}")

    # 优化器
    optimizer = torch.optim.Adam(head_params, lr=args.lr)  # 下层优化器 (head)
    mask_optimizer = torch.optim.SGD(prompt_params, lr=args.mask_lr)  # 上层优化器 (prompt tokens)
    scaler = GradScaler()

    # Training Loop
    best_acc = 0.0
    for epoch in range(args.epoch):
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch} Training", ncols=100)

        # 打印每个 epoch 的参数信息
        total_params, non_zero_params = count_total_and_nonzero_params(network)
        print(f"Epoch {epoch}: Total trainable parameters: {total_params}")
        print(f"Epoch {epoch}: Non-zero trainable parameters: {non_zero_params}")

        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mask_optimizer.zero_grad()

            # 下层优化 (更新 head)
            with autocast():
                logits = network(x)
                loss = F.cross_entropy(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # 上层优化 (更新 prompt tokens)
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
        prune_model(prompt_params + head_params, args.k)
        pruning_ratio = get_pruning_ratio(prompt_params + head_params)
        print(f"Epoch {epoch}: Pruning Ratio = {pruning_ratio:.2f}%")

        # 验证阶段
        # network.eval()
        # total_num, true_num = 0, 0
        # with torch.no_grad():
        #     pbar = tqdm(loaders['test'], total=len(loaders['test']), desc=f"Epoch {epoch} Testing", ncols=100)
        #     for x, y in pbar:
        #         x, y = x.to(device), y.to(device)
        #         outputs = network(x)
        #         total_num += y.size(0)
        #         true_num += outputs.argmax(dim=1).eq(y).sum().item()
        #         pbar.set_postfix_str(f"Acc: {100 * true_num / total_num:.2f}%")

        # acc = true_num / total_num
        # logger.add_scalar("test/accuracy", acc, epoch)

        # print(f"Epoch {epoch} - Validation: Total trainable parameters: {count_trainable_parameters(prompt_params + head_params)}")
        # print(f"Epoch {epoch} - Validation: Non-zero trainable parameters: {count_nonzero_trainable_parameters(prompt_params + head_params)}")

        # # 保存模型
        # torch.save(network.state_dict(), os.path.join(save_path, f"epoch_{epoch}.pth"))
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
        auc = roc_auc_score(y_true, y_prob) if num_classes == 2 else None  # 多分类任务不计算 AUC

        logger.add_scalar("test/acc", acc, epoch)
        logger.add_scalar("test/f1", f1, epoch)
        if auc is not None:
            logger.add_scalar("test/auc", auc, epoch)
        print(f"Epoch {epoch}: F1-score = {f1:.4f}, AUC = {auc:.4f}" if auc else f"Epoch {epoch}: F1-score = {f1:.4f}")


        # 保存最佳模型
        state_dict = {
            "network_dict": network.state_dict(),
            "optimizer_dict": optimizer.state_dict(),
            "epoch": epoch,
            "best_acc": best_acc,
        }
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            torch.save(network.state_dict(), os.path.join(save_path, "best.pth"))
        torch.save(network.state_dict(), os.path.join(save_path, "last.pth"))

    print(f"Training complete. Best accuracy: {100 * acc:.2f}%")
