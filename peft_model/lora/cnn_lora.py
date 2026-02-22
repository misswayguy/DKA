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

# Import your custom LoRA layers
from .lora_layers import Conv2d, Linear

class BackboneWithLoRA(nn.Module):
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
        super(BackboneWithLoRA, self).__init__()

        if backbone in ["resnet18", "resnet50"]:
            # Load ResNet model
            if backbone == "resnet18":
                base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            elif backbone == "resnet50":
                base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)

            # Freeze backbone parameters
            if freeze_backbone:
                for param in base_model.parameters():
                    param.requires_grad = False

            # Replace ResNet layers with LoRA-enabled layers
            self.layer1 = self._replace_lora(base_model.layer1, rank, lora_alpha, lora_dropout)
            self.layer2 = self._replace_lora(base_model.layer2, rank, lora_alpha, lora_dropout)
            self.layer3 = self._replace_lora(base_model.layer3, rank, lora_alpha, lora_dropout)
            self.layer4 = self._replace_lora(base_model.layer4, rank, lora_alpha, lora_dropout)

            self.conv1 = base_model.conv1
            self.bn1 = base_model.bn1
            self.relu = base_model.relu
            self.maxpool = base_model.maxpool
            self.avgpool = base_model.avgpool

            # Replace the classifier head
            in_features = base_model.fc.in_features
            self.fc = Linear(in_features, num_classes, r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        elif backbone in ["convnext_tiny", "convnext_base"]:
            # Load ConvNeXt model
            if backbone == "convnext_tiny":
                base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1 if pretrained else None)
            elif backbone == "convnext_base":
                base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1 if pretrained else None)

            # Freeze backbone parameters
            if freeze_backbone:
                for param in base_model.parameters():
                    param.requires_grad = False

            # Replace ConvNeXt layers with LoRA-enabled layers
            self.features = self._replace_lora_convnext(base_model.features, rank, lora_alpha, lora_dropout)

            # Replace the classifier head
            in_features = base_model.classifier[2].in_features
            self.classifier = Linear(in_features, num_classes, r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout)

        elif backbone in ["vgg16", "vgg19"]:
            # Load VGG model
            if backbone == "vgg16":
                base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1 if pretrained else None)
            elif backbone == "vgg19":
                base_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1 if pretrained else None)

            # Freeze backbone parameters
            if freeze_backbone:
                for param in base_model.features.parameters():
                    param.requires_grad = False

            # Replace VGG layers with LoRA-enabled layers
            self.features = self._replace_lora_vgg(base_model.features, rank, lora_alpha, lora_dropout)

            # Replace the classifier head
            in_features = base_model.classifier[0].in_features
            self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
            self.classifier = nn.Sequential(
                nn.Flatten(),
                Linear(in_features, num_classes, r=rank, lora_alpha=lora_alpha, lora_dropout=lora_dropout),
            )

        else:
            raise ValueError("Unsupported backbone! Please choose from ['resnet18', 'resnet50', 'convnext_tiny', 'convnext_base', 'vgg16', 'vgg19']")

    def _replace_lora(self, layer, rank, lora_alpha, lora_dropout):
        for block in layer:
            block.conv2 = Conv2d(
                in_channels=block.conv2.in_channels,
                out_channels=block.conv2.out_channels,
                kernel_size=block.conv2.kernel_size,
                stride=block.conv2.stride,
                padding=block.conv2.padding,
                r=rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                bias=block.conv2.bias is not None,
            )
        return layer

    def _replace_lora_convnext(self, features, rank, lora_alpha, lora_dropout):
        for module in features.modules():
            if isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                module = Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    r=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=module.bias is not None,
                )
        return features

    def _replace_lora_vgg(self, features, rank, lora_alpha, lora_dropout):
        for idx, module in enumerate(features):
            if isinstance(module, nn.Conv2d):
                kernel_size = module.kernel_size[0] if isinstance(module.kernel_size, tuple) else module.kernel_size
                features[idx] = Conv2d(
                    in_channels=module.in_channels,
                    out_channels=module.out_channels,
                    kernel_size=kernel_size,
                    stride=module.stride,
                    padding=module.padding,
                    r=rank,
                    lora_alpha=lora_alpha,
                    lora_dropout=lora_dropout,
                    bias=module.bias is not None,
                )
        return features

    def forward(self, x):
        if hasattr(self, "conv1"):  # ResNet
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
        elif hasattr(self, "features") and hasattr(self, "classifier"):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        elif hasattr(self, "stem"):
            x = self.features(x)
            x = torch.mean(x, dim=(2, 3))
            x = self.classifier(x)
        return x
