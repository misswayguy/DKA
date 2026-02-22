import torch
import torch.nn as nn
from torchvision.models import (
    resnet18, ResNet18_Weights,
    resnet50, ResNet50_Weights,
    convnext_tiny, ConvNeXt_Tiny_Weights,
    convnext_base, ConvNeXt_Base_Weights,
    vgg16, VGG16_Weights,
    vgg19, VGG19_Weights
)

# 定义 Adapter 模块
class Adapter(nn.Module):
    def __init__(self, in_channels, reduction=96):
        super(Adapter, self).__init__()
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1, bias=False),
            nn.ReLU()
        )

    def forward(self, x):
        return x + self.adapter(x)  # 残差连接

# 定义 Backbone + Adapter 模型
class BackboneWithAdapter(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, reduction=4, freeze_backbone=True, num_classes=10):
        super(BackboneWithAdapter, self).__init__()

        # 加载不同的 backbone
        if backbone == "resnet18":
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        elif backbone == "convnext_tiny":
            base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "convnext_base":
            base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "vgg16":
            base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "vgg19":
            base_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)
        else:
            raise ValueError("Unsupported backbone! Choose from ['resnet18', 'resnet50', 'convnext_tiny', 'convnext_base', 'vgg16', 'vgg19']")

        self.backbone_name = backbone

        if "resnet" in backbone:
            # ResNet 结构
            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.layer1 = self._add_adapters(base_model.layer1, reduction)
            self.layer2 = self._add_adapters(base_model.layer2, reduction)
            self.layer3 = self._add_adapters(base_model.layer3, reduction)
            self.layer4 = self._add_adapters(base_model.layer4, reduction)
            self.avgpool = base_model.avgpool
            in_features = base_model.fc.in_features

        elif "convnext" in backbone:
            # ConvNeXt 结构
            self.features = self._add_adapters_convnext(base_model.features, reduction)
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # ConvNeXt 固定的全局池化层
            in_features = base_model.classifier[2].in_features

        elif "vgg" in backbone:
            # VGG 结构
            self.features = self._add_adapters_vgg(base_model.features, reduction)
            self.avgpool = base_model.avgpool
            in_features = base_model.classifier[0].in_features

        # 分类头
        self.fc = nn.Linear(in_features, num_classes)

        # 冻结 Backbone
        if freeze_backbone:
            for name, param in self.named_parameters():
                if "adapter" not in name and "fc" not in name:  # 只训练 Adapter 和 fc 层
                    param.requires_grad = False

    def _add_adapters(self, layer, reduction):
        """
        为 ResNet 的每个残差块添加 Adapter 模块。
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

    def _add_adapters_convnext(self, layer, reduction):
        """
        为 ConvNeXt 的所有卷积层添加 Adapter 模块。
        """
        new_modules = []
        for module in layer:
        # 如果是 Conv2d，则添加 Adapter
            if isinstance(module, nn.Conv2d):
                new_module = nn.Sequential(
                module,  # 原始 Conv2d 模块
                Adapter(module.out_channels, reduction)  # 添加 Adapter 模块
                )
                new_modules.append(new_module)
            elif isinstance(module, nn.Sequential):
            # 如果是 Sequential，递归处理
                new_modules.append(self._add_adapters_convnext(module, reduction))
            else:
                new_modules.append(module)  # 保留其他模块

    # 返回修改后的层
        return nn.Sequential(*new_modules)


    def _add_adapters_vgg(self, layers, reduction):
        """
        为 VGG 的每个卷积层添加 Adapter 模块。
        """
        for module in layers:
            if isinstance(module, nn.Conv2d):
                module.adapter = Adapter(module.out_channels, reduction)
                original_forward = module.forward

                def forward_with_adapter(x, original_forward=original_forward, adapter=module.adapter):
                    x = original_forward(x)
                    return adapter(x)
                module.forward = forward_with_adapter
        return layers

    def forward(self, x):
        if "resnet" in self.backbone_name:
            # ResNet 前向传播
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
        elif "convnext" in self.backbone_name:
            # ConvNeXt 前向传播
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        elif "vgg" in self.backbone_name:
            # VGG 前向传播
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
        x = self.fc(x)  # 分类头
        return x
