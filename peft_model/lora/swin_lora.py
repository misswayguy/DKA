import torch
import torch.nn as nn
from torchvision.models import swin_b, swin_t, Swin_B_Weights, Swin_T_Weights
from .lora_layers import Linear  # LoRA 实现的 Linear

class SwinWithLoRA(nn.Module):
    def __init__(
        self,
        backbone="swin_t",
        pretrained=True,
        freeze_backbone=True,
        rank=4,
        lora_alpha=16,
        lora_dropout=0.1,
        num_classes=10,
    ):
        super(SwinWithLoRA, self).__init__()

        # 加载 Swin Transformer
        if backbone == "swin_b":
            self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "swin_t":
            self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 冻结主干网络参数
        if freeze_backbone:
            for param in self.swin.parameters():
                param.requires_grad = False

        # 遍历 Transformer Blocks，替换 MHSA 和 MLP 的线性层为 LoRA
        for module_name, module in self.swin.features.named_children():
            for block_name, block in module.named_children():
                if hasattr(block, "attn") and hasattr(block.attn, "qkv"):
                    # 替换注意力层的输出投影
                    block.attn.proj = Linear(
                        in_features=block.attn.proj.in_features,
                        out_features=block.attn.proj.out_features,
                        r=rank,
                        lora_alpha=lora_alpha,
                        lora_dropout=lora_dropout,
                    )

                if hasattr(block, "mlp"):
                    # 替换 MLP 层
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

        # 替换分类头
        swin_hidden_dim = self.swin.head.in_features
        self.swin.head = nn.Identity()  # 移除原始分类头
        self.classifier = nn.Linear(swin_hidden_dim, num_classes)

    def forward(self, x):
        x = self.swin(x)  # 提取特征
        x = self.classifier(x)  # 分类
        return x

# 测试代码
if __name__ == "__main__":
    # 测试 Swin-Tiny + LoRA
    swin_t_model = SwinWithLoRA(backbone="swin_t", num_classes=3)
    x = torch.randn(4, 3, 224, 224)  # 输入数据
    output = swin_t_model(x)
    print("Swin-T output shape:", output.shape)

    # 测试 Swin-Base + LoRA
    swin_b_model = SwinWithLoRA(backbone="swin_b", num_classes=3)
    output = swin_b_model(x)
    print("Swin-B output shape:", output.shape)
