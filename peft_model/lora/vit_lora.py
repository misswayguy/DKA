import torch
import torch.nn as nn
from torchvision.models import vit_b_16, ViT_B_16_Weights, vit_l_16, ViT_L_16_Weights
from .lora_layers import Linear  # LoRA 实现的 Linear

class ViTWithLoRA(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_b_16",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        rank: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_classes: int = 3,  # 修改为三分类
    ):
        super(ViTWithLoRA, self).__init__()

        # 加载预训练 ViT 模型
        if backbone == "vit_b_16":
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "vit_l_16":
            self.vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("Unsupported model name! Use 'vit_b_16' or 'vit_l_16'.")

        # 冻结 Backbone 权重（除 LoRA 和分类头外）
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # 替换 Self-Attention 中的投影层为 LoRA
        for block in self.vit.encoder.layers:
            mha = block.self_attention
            mha.out_proj = Linear(
                in_features=mha.out_proj.in_features,
                out_features=mha.out_proj.out_features,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

        # 替换 FFN 的线性层
        for block in self.vit.encoder.layers:
            block.mlp[0] = Linear(
                in_features=block.mlp[0].in_features,
                out_features=block.mlp[0].out_features,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )
            block.mlp[3] = Linear(
                in_features=block.mlp[3].in_features,
                out_features=block.mlp[3].out_features,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
            )

        # 替换分类头，支持三分类
        #self.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
        hidden_dim = self.vit.heads.head.in_features
        self.vit.heads = nn.Sequential(
            nn.Linear(hidden_dim, num_classes)  # 新的分类头
        )

    def forward(self, x):
        """
        前向传播
        """
    # 获取 ViT 模型的输出
        x = self.vit(x)  # 输出形状可能是 (batch_size, seq_length, hidden_dim) 或 (batch_size, hidden_dim)
        return x

