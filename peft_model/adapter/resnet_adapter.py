import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights

# 定义 Adapter 模块
class Adapter(nn.Module):
    def __init__(self, in_channels, reduction=64):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.adapter(x)  # 残差连接

# 定义 ResNet+Adapter 网络
class ResNetWithAdapter(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, reduction=4, freeze_backbone=True, num_classes=10):
        super(ResNetWithAdapter, self).__init__()
        
        # 加载预训练模型
        if backbone == "resnet18":
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            raise ValueError("Unsupported backbone!")

        # 保留主干网络层
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        # 添加 Adapter 模块到每个卷积层后
        self.layer1 = self._add_adapters(base_model.layer1, reduction)
        self.layer2 = self._add_adapters(base_model.layer2, reduction)
        self.layer3 = self._add_adapters(base_model.layer3, reduction)
        self.layer4 = self._add_adapters(base_model.layer4, reduction)

        self.avgpool = base_model.avgpool

        # 重新定义分类头
        in_features = base_model.fc.in_features  # 获取 fc 层输入特征数
        self.fc = nn.Linear(in_features, num_classes)

        # 冻结 Backbone（包括卷积层和 BN 层），只训练 Adapter 和分类头
        if freeze_backbone:
            for name, param in self.named_parameters():
                if "adapter" not in name and "fc" not in name:  # 只训练 Adapter 和 fc 层
                    param.requires_grad = False

    def _add_adapters(self, layer, reduction):
        """
        为网络的每个残差块的卷积层后面添加 Adapter 模块。
        """
        for block in layer:
            block.adapter = Adapter(block.conv2.out_channels, reduction)
            # 修改残差块的 forward 方法，串联 Adapter
            original_forward = block.forward

            def forward_with_adapter(x, original_forward=original_forward, adapter=block.adapter):
                x = original_forward(x)  # 原始 forward 计算
                return adapter(x)        # 添加 Adapter
            block.forward = forward_with_adapter
        return layer

    def forward(self, x):
        # 主干网络前向传播
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)  # 展平特征
        x = self.fc(x)           # 分类头
        return x
