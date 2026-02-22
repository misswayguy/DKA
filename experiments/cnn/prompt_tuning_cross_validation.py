import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse

import sys
sys.path.append(".")
from tools.misc import gen_folder_name, set_seed
from cfg import *

from sklearn.metrics import f1_score, recall_score


from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略损坏的图片


from peft_model.prompt.resnet_prompt import ResNetWithPrompt
from peft_model.prompt.vit_prompt import ViTWithPrompt  # 导入 Prompt 模型
from peft_model.prompt.cnn_prompt import ConvNeXtWithPrompt, VGGWithPrompt
from peft_model.prompt.swin_prompt import SwinWithPrompt



# 数据加载函数
def prepare_custom_data(data_path, preprocess):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
    test_dataset = datasets.ImageFolder(root=test_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    return {"train": train_loader, "test": test_loader}, class_names


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)



class PromptTuningConfig:
    def __init__(self, num_virtual_tokens, token_dim, inference_mode=False):
        self.num_virtual_tokens = num_virtual_tokens
        self.token_dim = token_dim
        self.inference_mode = inference_mode


if __name__ == '__main__':
    # 解析参数
    p = argparse.ArgumentParser()
    p.add_argument('--network', choices=["resnet18_prompt", "convnext_tiny_prompt", "convnext_base_prompt", "vgg16_prompt", "vgg19_prompt", 
                                        "swin_tiny_prompt", "swin_base_prompt", 
                                        "vit_b16_prompt_11th","vit_b16_prompt_10th","vit_b16_prompt_9th","vit_b16_prompt_8th","vit_b16_prompt_7th","vit_b16_prompt_0th",
                                         "vit_b16_prompt_6th", "vit_b16_prompt_5th", "vit_b16_prompt_4th", "vit_b16_prompt_3th", "vit_b16_prompt_2th","vit_b16_prompt_1th",
                                        "vit_b16_prompt", "vit_l16_prompt"], required=True)
    # p.add_argument('--dataset', choices=["covid_full", "covid"], required=True)
    p.add_argument('--dataset', type=str, required=True, help="Path to dataset")
    p.add_argument('--epoch', type=int, default=80)
    p.add_argument('--lr', type=float, default=0.0001)
    p.add_argument('--virtual_tokens', type=int, default=50, help="Number of virtual tokens for Prompt")
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0, cuda:1, cuda:2, cuda:3, cuda:4, cuda:5)')
    args = p.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)



    # 生成实验路径
    exp = "/mnt/data/lsy/ZZQ/prompt_tuning"
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
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # 修改为你给的代码中的处理方式
    ])

    # 加载数据集
    print(f"Loading dataset: {args.dataset}")
    loaders, class_names = prepare_custom_data(args.dataset, preprocess)  # 使用prepare_custom_data加载数据
    num_classes = len(class_names)

    # 加载网络
    if args.network == "resnet18_prompt":
        network = ResNetWithPrompt(
            num_virtual_tokens=args.virtual_tokens,
            num_classes=num_classes,
            freeze_backbone=True
        ).to(device)
    elif args.network == "convnext_tiny_prompt":
        network = ConvNeXtWithPrompt(backbone="convnext_tiny", num_virtual_tokens=args.virtual_tokens, 
                                     num_classes=num_classes, freeze_backbone=True).to(device)
    elif args.network == "convnext_base_prompt":
        network = ConvNeXtWithPrompt(backbone="convnext_base", num_virtual_tokens=args.virtual_tokens, 
                                     num_classes=num_classes, freeze_backbone=True).to(device)
    elif args.network == "vgg16_prompt":
        network = VGGWithPrompt(backbone="vgg16", num_virtual_tokens=args.virtual_tokens, 
                                     num_classes=num_classes, freeze_backbone=True).to(device) 
    elif args.network == "vgg19_prompt":
        network = VGGWithPrompt(backbone="vgg19", num_virtual_tokens=args.virtual_tokens, 
                                     num_classes=num_classes, freeze_backbone=True).to(device) 
    elif args.network == "vit_b16_prompt":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[],
        ).to(device)
    elif args.network == "vit_b16_prompt_11th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[11],
        ).to(device)
    elif args.network == "vit_b16_prompt_10th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[10],
        ).to(device)
    elif args.network == "vit_b16_prompt_9th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[9],
        ).to(device)
    elif args.network == "vit_b16_prompt_8th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[8],
        ).to(device)
    elif args.network == "vit_b16_prompt_7th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[7],
        ).to(device)
    elif args.network == "vit_b16_prompt_6th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[6],
        ).to(device)
    elif args.network == "vit_b16_prompt_5th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[5],
        ).to(device)
    elif args.network == "vit_b16_prompt_4th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[4],
        ).to(device)
    elif args.network == "vit_b16_prompt_3th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[3],
        ).to(device)
    elif args.network == "vit_b16_prompt_2th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[2],
        ).to(device)
    elif args.network == "vit_b16_prompt_1th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[1],
        ).to(device)
    elif args.network == "vit_b16_prompt_0th":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=768,  # ViT Base 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Base 模型
        network = ViTWithPrompt(
            backbone="vit_b16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
            selected_blocks=[0],
        ).to(device)
    elif args.network == "vit_l16_prompt":
        # 配置 PromptTuningConfig
        config = PromptTuningConfig(
            num_virtual_tokens=args.virtual_tokens,
            token_dim=1024,  # ViT Large 的嵌入维度
            inference_mode=False,
        )
        # 初始化 ViT Large 模型
        network = ViTWithPrompt(
            backbone="vit_l16",
            num_prompts=config.num_virtual_tokens,
            prompt_dim=config.token_dim,
            num_classes=num_classes,
            freeze_backbone=True,
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
    

    trainable_params = count_trainable_parameters(network)
    print(f"Trainable parameters: {trainable_params}")

    # 优化器
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)

    # 混合精度训练
    scaler = GradScaler()

    # 训练和测试循环
    best_acc = 0.0
    for epoch in range(args.epoch):
        # 训练阶段
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0
        pbar = tqdm(loaders['train'], total=len(loaders['train']), desc=f"Epoch {epoch} Training", ncols=100)
        for x, y in pbar:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            with autocast():
                outputs = network(x)
                loss = F.cross_entropy(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_num += y.size(0)
            true_num += outputs.argmax(dim=1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)
            pbar.set_postfix_str(f"Acc: {100 * true_num / total_num:.2f}%, Loss: {loss_sum / total_num:.4f}")
        
        scheduler.step()
        logger.add_scalar("train/accuracy", true_num / total_num, epoch)
        logger.add_scalar("train/loss", loss_sum / total_num, epoch)

        # 测试阶段
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
                # pbar.set_postfix_str(f"Acc {100 * acc:.2f}%")
                f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
                sensitivity = recall_score(y_true, y_pred, average='macro', zero_division=0)
                pbar.set_postfix_str(f"Acc {100 * acc:.2f}%, F1 {f1:.3f}, Sen {sensitivity:.3f}")


        # f1 = f1_score(y_true, y_pred, average='weighted')  # 加权平均 F1
        # auc = roc_auc_score(y_true, y_prob) if num_classes == 2 else None  # 多分类任务不计算 AUC11

        logger.add_scalar("test/acc", acc, epoch)
        # logger.add_scalar("test/f1", f1, epoch)
        # if auc is not None:
        #     logger.add_scalar("test/auc", auc, epoch)
        # print(f"Epoch {epoch}: F1-score = {f1:.4f}, AUC = {auc:.4f}" if auc else f"Epoch {epoch}: F1-score = {f1:.4f}")


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

    print(f"Training complete. Best accuracy: {100 * best_acc:.2f}%")
