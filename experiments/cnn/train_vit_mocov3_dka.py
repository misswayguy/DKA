import os
import time
import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from tqdm import tqdm
from PIL import ImageFile

import timm  # ViT backbone
from safetensors.torch import load_file  # 读取 .safetensors 权重

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略坏图


# ------------------------
# 1. 随机种子
# ------------------------
def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------
# 2. 数据加载
# ------------------------
def prepare_custom_data(train_path, test_path, preprocess, batch_size=64, num_workers=4):
    train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
    test_dataset = datasets.ImageFolder(root=test_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    class_names = train_dataset.classes
    return {"train": train_loader, "test": test_loader}, class_names


# ------------------------
# 3. 你的 DKA Adapter 模块
# ------------------------
class Adapter(nn.Module):
    def __init__(self, input_dim, middle_dim=None, reduction=0.25):
        super(Adapter, self).__init__()

        # 线性降维，等效于 1x1 卷积
        self.adapter_down = nn.Linear(input_dim, middle_dim, bias=False)
        self.adapter_up = nn.Linear(middle_dim, input_dim, bias=False)

        # 51×51 深度可分离卷积（大核）
        self.conv1 = nn.Conv2d(
            middle_dim, middle_dim,
            kernel_size=51, stride=1, padding=25,
            groups=middle_dim, bias=False
        )
        # 5×5 深度可分离卷积（小核）
        self.conv2 = nn.Conv2d(
            middle_dim, middle_dim,
            kernel_size=5, stride=1, padding=2,
            groups=middle_dim, bias=False
        )

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        """
        x: (B, N, C)  —— ViT block 的输入/输出 token
        N: 1 (cls) + H*W
        """
        B, N, C = x.shape  # (batch, seq_len, embed_dim)

        # Step 1: 线性降维
        x_down = self.adapter_down(x)  # (B, N, reduced_dim)
        x_down = self.act(x_down)

        # Step 2: 变形适配 Conv2D
        H = W = int((N - 1) ** 0.5)  # 假设 cls_token 之外的是正方形 patch grid
        x_patch = x_down[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # (B, C', H, W)

        # Step 3: 大核 + 小核 并联卷积
        conv1_out = self.conv1(x_patch)  # 51×51
        conv2_out = self.conv2(x_patch)  # 5×5
        x_patch = conv1_out + conv2_out  # 逐元素相加（DKA）

        # Step 4: 变回 Transformer token 格式
        x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # (B, N-1, reduced_dim)

        # Step 5: 处理 cls_token
        x_cls = x_down[:, :1]  # (B, 1, reduced_dim)

        # Step 6: 拼接
        x_down = torch.cat([x_cls, x_patch], dim=1)  # (B, N, reduced_dim)
        x_down = self.act(x_down)
        x_down = self.dropout(x_down)

        # Step 7: 线性升维 + 残差
        x_up = self.adapter_up(x_down)  # (B, N, embed_dim)

        return x + x_up  # 残差连接，保持维度不变


# ------------------------
# 4. 基于 timm ViT + 本地 MoCo v3 权重 的模型封装
# ------------------------
class MoCoV3ViTWithAdapterDKA(nn.Module):
    def __init__(
        self,
        num_classes: int,
        middle_dim: int = 16,
        reduction: float = 0.25,
        selected_layers=None,
        freeze_backbone: bool = True,
        mocov3_ckpt_path: str = None,   # 本地权重路径
    ):
        super().__init__()

        # 1) 建一个“空”的 ViT-B/16（不加载预训练）
        self.vit = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False,
        )
        embed_dim = self.vit.embed_dim  # 768 for ViT-B

        # 2) 从本地 .safetensors 加载 MoCo v3 权重
        if mocov3_ckpt_path is None:
            raise ValueError("必须提供 mocov3_ckpt_path，本地 MoCo v3 权重路径")

        print(f"[MoCo-v3] Loading checkpoint from: {mocov3_ckpt_path}")
        state = load_file(mocov3_ckpt_path)  # 直接得到 state_dict: key -> tensor

        # 保险起见：如果以后换成 .pth，这里也兼容一下
        if isinstance(state, dict):
            if "state_dict" in state:
                state = state["state_dict"]
            elif "model" in state:
                state = state["model"]

        missing, unexpected = self.vit.load_state_dict(state, strict=False)
        print("[MoCo-v3] missing keys:", missing)
        print("[MoCo-v3] unexpected keys:", unexpected)
        print("[MoCo-v3] 自监督权重已加载到 ViT backbone。")

        # 3) 替换分类头（head），适配当前任务类别数
        self.vit.head = nn.Linear(embed_dim, num_classes)

        # 4) 给指定层插入 Adapter（DKA 模块）
        if selected_layers is None:
            selected_layers = [11]  # 默认第 11 层（最后一层）

        for idx, block in enumerate(self.vit.blocks):
            if idx in selected_layers:
                block.adapter = Adapter(embed_dim, middle_dim=middle_dim, reduction=reduction)
                original_forward = block.forward

                def forward_with_adapter(x, original_forward=original_forward, adapter=block.adapter):
                    x = original_forward(x)  # 原始 ViT Block
                    x = adapter(x)           # 通过 DKA Adapter
                    return x

                block.forward = forward_with_adapter

        # 5) 冻结 backbone（只训练 head & adapter）
        if freeze_backbone:
            for name, param in self.vit.named_parameters():
                param.requires_grad = False

            for name, param in self.vit.named_parameters():
                if "adapter" in name or "head" in name:
                    param.requires_grad = True

    def forward(self, x):
        return self.vit(x)


# ------------------------
# 5. 统计训练参数 + 获取 adapter/head 参数
# ------------------------
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_adapter_and_head_params(model):
    adapter_params = []
    head_params = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "adapter" in name:
            adapter_params.append(param)
        elif "head" in name:
            head_params.append(param)

    return adapter_params, head_params


# ------------------------
# 6. 主程序：MoCo v3 + DKA + 冻结 backbone
# ------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_dataset", type=str, required=True, help="Path to train dataset root")
    parser.add_argument("--test_dataset", type=str, required=True, help="Path to test dataset root")
    parser.add_argument("--mocov3_ckpt", type=str, required=True,
                        help="本地 MoCo v3 ViT-B 权重路径（.safetensors）")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr_head", type=float, default=1e-4)      # head 学习率
    parser.add_argument("--lr_adapter", type=float, default=1e-3)   # adapter 学习率
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--middle_dim", type=int, default=16, help="Adapter bottleneck dim")
    parser.add_argument("--selected_layers", type=str, default="11",
                        help="逗号分隔的层号，例如 '8,9,10,11'")

    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    print(f"Loading train dataset from {args.train_dataset}")
    print(f"Loading test  dataset from {args.test_dataset}")
    loaders, class_names = prepare_custom_data(
        args.train_dataset, args.test_dataset, preprocess,
        batch_size=args.batch_size
    )

    num_classes = len(class_names)
    print("Classes:", class_names)

    # 解析 selected_layers
    selected_layers = [int(x) for x in args.selected_layers.split(",")]

    # 构建模型（MoCo v3 ViT-B + DKA Adapter）
    model = MoCoV3ViTWithAdapterDKA(
        num_classes=num_classes,
        middle_dim=args.middle_dim,
        selected_layers=selected_layers,
        freeze_backbone=True,
        mocov3_ckpt_path=args.mocov3_ckpt,
    ).to(device)

    print(f"Total trainable parameters (DKA + head only): {count_trainable_parameters(model)}")

    # 获取 Adapter 和 Head 参数，分别设置不同学习率
    adapter_params, head_params = get_adapter_and_head_params(model)

    optimizer = torch.optim.Adam([
        {"params": head_params, "lr": args.lr_head},
        {"params": adapter_params, "lr": args.lr_adapter},
    ])
    scaler = GradScaler()

    best_acc = 0.0

    # ----------------- 训练 + 测试循环 -----------------
    for epoch in range(args.epochs):
        # -------- 训练阶段 --------
        model.train()
        total_num, true_num, loss_sum = 0, 0, 0.0
        pbar = tqdm(loaders["train"], desc=f"Epoch {epoch+1} Training", ncols=100)

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                logits = model(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

            total_num += y.size(0)
            true_num += logits.argmax(1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)
            pbar.set_postfix_str(
                f"Acc: {100.0 * true_num / total_num:.2f}%, "
                f"Loss: {loss_sum / total_num:.4f}"
            )

        train_acc = true_num / total_num
        print(f"Epoch {epoch+1} - Train Loss: {loss_sum/total_num:.4f}, "
              f"Train Acc: {train_acc*100:.2f}%")

        # -------- latency / memory profiling（可选）--------
        model.eval()
        print("Profiling inference latency and memory usage...")

        test_iter = iter(loaders["test"])
        x_test, y_test = next(test_iter)
        x_test = x_test.to(device)

        # 预热
        with torch.no_grad():
            for _ in range(5):
                _ = model(x_test)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        with torch.no_grad():
            _ = model(x_test)
        torch.cuda.synchronize()
        end_time = time.time()

        latency = (end_time - start_time) * 1000.0  # ms
        peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

        print(f"Inference latency: {latency:.2f} ms | Peak memory: {peak_memory:.2f} MB")

        # -------- 整个测试集计算 Accuracy --------
        total_num, true_num = 0, 0
        with torch.no_grad():
            for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
                x, y = x.to(device), y.to(device)
                outputs = model(x)
                preds = outputs.argmax(1)

                total_num += y.size(0)
                true_num += preds.eq(y).sum().item()

        test_acc = true_num / total_num
        print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), "best_vit_mocov3_dka.pth")
            print(f"*** New best model saved (Acc = {best_acc*100:.2f}%) ***")

    print(f"Training complete. Best Accuracy: {best_acc*100:.2f}%")


if __name__ == "__main__":
    main()
