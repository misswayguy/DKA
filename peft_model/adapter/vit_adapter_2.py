import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights


# Adapter 模块
class Adapter(nn.Module):
    def __init__(self, input_dim, middle_dim=None, reduction=16):
        super(Adapter, self).__init__()
        if middle_dim is None:
            middle_dim = max(1, input_dim // reduction)

        self.adapter = nn.Sequential(
            nn.Linear(input_dim, middle_dim, bias=False),
            nn.ReLU(),
            nn.Linear(middle_dim, input_dim, bias=False),
        )

    def forward(self, x):
        return x + self.adapter(x)  # 残差连接
    


# Vision Transformer 模型（带 Adapter）
class ViTWithAdapter_2(nn.Module):
    def __init__(self, backbone="vit_b16", pretrained=True, reduction=16, middle_dim=None, freeze_backbone=True, num_classes=10):
        super(ViTWithAdapter_2, self).__init__()

        # 加载预训练的 ViT 模型
        if backbone == "vit_b16":
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dim = 768
        elif backbone == "vit_l16":
            self.vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dim = 1024
        else:
            raise ValueError("Unsupported backbone. Use 'vit_b16' or 'vit_l16'.")

        # 冻结 backbone 的参数
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # 为每个 Encoder Block 添加 Adapter
        for idx, block in enumerate(self.vit.encoder.layers):
            # 添加 Adapter 到 Multi-Head Attention 后
            block.attention_adapter = Adapter(hidden_dim, middle_dim=middle_dim, reduction=reduction)
            original_attention_forward = block.self_attention.forward

            def attention_with_adapter(x, original_forward=original_attention_forward, adapter=block.attention_adapter, **kwargs):
                residual = x  # Attention 的残差连接
                x = original_forward(x, **kwargs)  # 原始 Attention
                x = adapter(x) + residual  # Adapter + 残差连接
                return x

            block.self_attention.forward = attention_with_adapter

            # 添加 Adapter 到 Feed-Forward Network（MLP）后
            block.ffn_adapter = Adapter(hidden_dim, middle_dim=middle_dim, reduction=reduction)
            original_ffn_forward = block.mlp.forward

            def ffn_with_adapter(x, original_forward=original_ffn_forward, adapter=block.ffn_adapter):
                residual = x  # FFN 的残差连接
                x = original_forward(x)  # 原始 FFN
                x = adapter(x) + residual  # Adapter + 残差连接
                return x

            block.mlp.forward = ffn_with_adapter

        # 替换分类头
        self.vit.heads.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x


# 测试更新后的模型
if __name__ == "__main__":
    # 测试 ViT-B/16 + Adapter
    vit_b_model = ViTWithAdapter_2(backbone="vit_b16", num_classes=10)
    x = torch.randn(4, 3, 224, 224)  # 输入数据
    output = vit_b_model(x)
    print("ViT-B/16 输出维度:", output.shape)

    # 测试 ViT-L/16 + Adapter
    vit_l_model = ViTWithAdapter_2(backbone="vit_l16", num_classes=10)
    output = vit_l_model(x)
    print("ViT-L/16 输出维度:", output.shape)
