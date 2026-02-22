import torch
import torch.nn as nn
from torchvision.models import convnext_tiny, convnext_base, ConvNeXt_Tiny_Weights, ConvNeXt_Base_Weights

# class Adapter(nn.Module):
#     def __init__(self, in_channels,middle_dim=None, reduction=0.25):
#         super(Adapter, self).__init__()
#         if in_channels * reduction == 0:
#             raise ValueError(f"Reduction factor {reduction} is too large for input channels {in_channels}")

#         self.adapter = nn.Sequential(
#             #nn.Conv2d(in_channels, int(in_channels * reduction), kernel_size=1, bias=False),
#             nn.Conv2d(in_channels, middle_dim, kernel_size=1, bias=False),
#             nn.ReLU(),
#             #nn.Conv2d(max(1, int(in_channels * reduction)), in_channels, kernel_size=1, bias=False),
#             nn.Conv2d(middle_dim, in_channels, kernel_size=1, bias=False),
#             nn.ReLU()
#         )

#     def forward(self, x):
#         return x + self.adapter(x)  # Residual connection

class Adapter(nn.Module):
    def __init__(self, in_channels, middle_dim=None, reduction=0.25):
        super(Adapter, self).__init__()
        
        # 计算降维通道数
        middle_dim = middle_dim or int(in_channels * reduction)
        if middle_dim < 1:
            raise ValueError(f"Reduction {reduction} is too large for in_channels {in_channels}")

        # 1x1 线性降维
        self.adapter_down = nn.Conv2d(in_channels, middle_dim, kernel_size=1, bias=False)
        self.adapter_up = nn.Conv2d(middle_dim, in_channels, kernel_size=1, bias=False)

        # 31×31 和 5×5 深度可分离卷积
        self.conv1 = nn.Conv2d(middle_dim, middle_dim, kernel_size=11, stride=1, padding=5, groups=middle_dim, bias=False)
        self.conv2 = nn.Conv2d(middle_dim, middle_dim, kernel_size=5, stride=1, padding=2, groups=middle_dim, bias=False)

        # 归一化 + GELU 激活 + Dropout
        self.norm = nn.BatchNorm2d(middle_dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        # Step 1: 线性降维
        x_down = self.adapter_down(x)
        x_down = self.act(x_down)

        # Step 2: 并联 31x31 和 5x5 卷积
        conv1_out = self.conv1(x_down)
        conv2_out = self.conv2(x_down)
        x_patch = conv1_out + conv2_out  # 逐元素相加

        # Step 3: 归一化 + 激活
        x_patch = self.norm(x_patch)
        x_patch = self.act(x_patch)
        x_patch = self.dropout(x_patch)

        # Step 4: 线性升维 + 残差连接
        x_up = self.adapter_up(x_patch)

        return x + x_up  # 残差连接


class ConvNeXtWithAdapter(nn.Module):
    def __init__(self, backbone="convnext_tiny", pretrained=True, middle_dim=None, reduction=4, freeze_backbone=True, num_classes=10):
        super(ConvNeXtWithAdapter, self).__init__()

        # Load ConvNeXt backbone
        if backbone == "convnext_tiny":
            self.backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "convnext_base":
            self.backbone = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Add Adapters to ConvNeXt features
        self.features = self._add_adapters(self.backbone.features,middle_dim, reduction)

        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        in_features = self.backbone.classifier[2].in_features
        self.classifier = nn.Linear(in_features, num_classes)

        # Freeze backbone if required
        if freeze_backbone:
            for name, param in self.named_parameters():
                if "adapter" not in name and "classifier" not in name:  # Only train adapters and classifier
                    param.requires_grad = False

    def _add_adapters(self, layer,middle_dim, reduction):
        """Add Adapter modules to ConvNeXt layers."""
        for name, module in layer.named_children():
            if isinstance(module, nn.Conv2d):
                # Add Adapter to Conv2d layers
                layer[int(name)] = nn.Sequential(
                    module,  # Original Conv2d module
                    Adapter(module.out_channels,middle_dim, reduction)  # Add Adapter
                )
            elif isinstance(module, nn.Sequential):
                # Recursively add Adapters for nested Sequential modules
                self._add_adapters(module,middle_dim, reduction)
        return layer

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

# Test code
if __name__ == "__main__":
    model = ConvNeXtWithAdapter(backbone="convnext_tiny", reduction=96, num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print("Output shape:", output.shape)
