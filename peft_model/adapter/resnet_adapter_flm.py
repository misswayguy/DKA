import torch
import torch.nn as nn
from torchvision.models import resnet18, resnet50, ResNet18_Weights, ResNet50_Weights

class Adapter(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        hidden_dim = in_channels // reduction
        self.adapter = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, in_channels, kernel_size=1),
        )
    
    def forward(self, x):
        return x + self.adapter(x)  # 残差连接


class ResNetWithAdapter_FLM(nn.Module):
    def __init__(self, backbone="resnet18", pretrained=True, reduction=4, freeze_backbone=True, 
                 num_classes=10, freeze_fc=False):
        super().__init__()
        
        # 加载主干网络
        if backbone == "resnet18":
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        elif backbone == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # 冻结主干网络
        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        # 替换残差块的 conv2 层为 Adapter
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = self._replace_layer_with_adapter(base_model.layer1, reduction)
        self.layer2 = self._replace_layer_with_adapter(base_model.layer2, reduction)
        self.layer3 = self._replace_layer_with_adapter(base_model.layer3, reduction)
        self.layer4 = self._replace_layer_with_adapter(base_model.layer4, reduction)
        
        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)
        
        # 冻结最后一层 fc（如果指定）
        if freeze_fc:
            for param in self.fc.parameters():
                param.requires_grad = False

    def _replace_layer_with_adapter(self, layer, reduction):
        for block in layer:
            block.conv2 = Adapter(block.conv2.in_channels, reduction)
        return layer

    def forward(self, x):
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
        x = self.fc(x)
        return x
