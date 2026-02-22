import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights


class Adapter(nn.Module):
    def __init__(self, input_dim, middle_dim=None, reduction=16):
        super(Adapter, self).__init__()

        # 默认中间层维度
        middle_dim = middle_dim or input_dim // reduction
        if middle_dim < 1:
            raise ValueError(f"Invalid middle_dim {middle_dim}, must be >= 1.")

        # 可训练控制因子 alpha
        self.alpha = nn.Parameter(torch.tensor(0.1))  # 初始化为 0.1
        # self.activation = nn.Softplus()  # 保证 alpha 非负，避免负贡献
        self.activation = nn.Sigmoid()

        # Adapter 模块
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, middle_dim, bias=False),
            nn.ReLU(),
            nn.Linear(middle_dim, input_dim, bias=False),
        )

    def forward(self, x):
        # 使用控制因子 alpha 调节 Adapter 输出的贡献
        #return x + self.activation(self.alpha) * self.adapter(x)
        return (1 - self.activation(self.alpha)) * x + self.activation(self.alpha) * self.adapter(x)


class ViTWithAdapter_alpha(nn.Module):
    def __init__(
        self,
        backbone="vit_b16",
        pretrained=True,
        reduction=16,
        middle_dim=None,
        freeze_backbone=True,
        num_classes=10,
        selected_layers=None,
    ):
        super(ViTWithAdapter_alpha, self).__init__()

        # 加载预训练 ViT 模型
        if backbone == "vit_b16":
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dim = 768
        elif backbone == "vit_l16":
            self.vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dim = 1024
        else:
            raise ValueError("Unsupported backbone. Use 'vit_b16' or 'vit_l16'.")

        # 冻结主干参数（可选）
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # 默认在最后一层添加 Adapter
        if selected_layers is None:
            selected_layers = [11]  # 默认只在第 11 层

        # 为选定的 Transformer Blocks 添加 Adapter
        for idx, block in enumerate(self.vit.encoder.layers):
            if idx in selected_layers:
                block.adapter = Adapter(hidden_dim, middle_dim=middle_dim, reduction=reduction)
                original_forward = block.forward

                def forward_with_adapter(x, original_forward=original_forward, adapter=block.adapter):
                    x = original_forward(x)  # 原始 Transformer 的前向传播
                    x = adapter(x)  # 通过 Adapter 调节输出
                    return x

                block.forward = forward_with_adapter

        # 替换分类头
        self.vit.heads.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        return self.vit(x)


# 测试代码
if __name__ == "__main__":
    # 测试 ViT-B/16 + Adapter
    vit_b_model = ViTWithAdapter_alpha(
        backbone="vit_b16",
        num_classes=10,
        reduction=16,
        middle_dim=5,  # 设置中间层维度为 5
        selected_layers=list(range(10, 12)),  # 为第 10 和第 11 层添加 Adapter
    )
    x = torch.randn(4, 3, 224, 224)  # 输入数据
    output = vit_b_model(x)
    print("ViT-B/16 output shape:", output.shape)

    # 测试 ViT-L/16 + Adapter
    vit_l_model = ViTWithAdapter_alpha(
        backbone="vit_l16",
        num_classes=10,
        reduction=32,
        middle_dim=10,  # 设置中间层维度为 10
        selected_layers=[11],  # 仅为第 11 层添加 Adapter
    )
    output = vit_l_model(x)
    print("ViT-L/16 output shape:", output.shape)
