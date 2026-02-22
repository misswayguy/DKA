import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

# 导入你自定义的 LoRA 模块
from .lora_layers import Conv2d, Linear


# 定义 ResNet + LoRA 模型
class ResNetWithLoRA(nn.Module):
    def __init__(
        self,
        backbone: str = "resnet18",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        rank: int = 4,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
        num_classes: int = 10,
    ):
        super(ResNetWithLoRA, self).__init__()

        # 加载预训练 ResNet 模型
        if backbone == "resnet18":
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            raise ValueError("Unsupported backbone!")

        # 冻结 Backbone 权重
        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        # 替换残差块中的 conv2 层为 LoRA 卷积层
        self.layer1 = self._replace_lora(base_model.layer1, rank, lora_alpha, lora_dropout)
        self.layer2 = self._replace_lora(base_model.layer2, rank, lora_alpha, lora_dropout)
        self.layer3 = self._replace_lora(base_model.layer3, rank, lora_alpha, lora_dropout)
        self.layer4 = self._replace_lora(base_model.layer4, rank, lora_alpha, lora_dropout)

        # 保留 ResNet 主干网络其他部分
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool
        self.avgpool = base_model.avgpool

        # 替换分类头 fc 层为 LoRA Linear 层
        in_features = base_model.fc.in_features
        self.fc = Linear(in_features, num_classes, r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

    def _replace_lora(self, layer, rank, lora_alpha, lora_dropout):
        """
        替换 ResNet 残差块中的 conv2 层为 LoRA 卷积层
        """
        for block in layer:
            block.conv2 = Conv2d(
                in_channels=block.conv2.in_channels,
                out_channels=block.conv2.out_channels,
                #kernel_size=block.conv2.kernel_size,
                kernel_size=block.conv2.kernel_size[0],
                stride=block.conv2.stride,
                padding=block.conv2.padding,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=block.conv2.bias is not None,
            )
        return layer

    def forward(self, x):
        """
        前向传播
        """
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)  # LoRA Linear 层
        return x
