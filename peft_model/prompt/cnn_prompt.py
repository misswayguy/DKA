import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights, resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights, convnext_base, ConvNeXt_Base_Weights
from torchvision.models import vgg16, VGG16_Weights, vgg19, VGG19_Weights
# 定义 PromptEmbedding 模块
class PromptEmbedding(nn.Module):
    def __init__(self, num_virtual_tokens, token_dim, reduction=4):
        """
        PromptEmbedding: 可学习的虚拟 token 转换为特征空间。
        Args:
            num_virtual_tokens (int): 虚拟 token 的数量。
            token_dim (int): 特征维度。
            reduction (int): 缩减维度的比例。
        """
        super(PromptEmbedding, self).__init__()
        self.num_virtual_tokens = num_virtual_tokens
        self.token_dim = token_dim

        # 可学习的 Embedding 层
        self.embedding = nn.Embedding(num_virtual_tokens, token_dim)

        # 映射到特征空间的简单网络
        self.mapping = nn.Sequential(
            nn.Linear(token_dim, token_dim // reduction),
            nn.ReLU(),
            nn.Linear(token_dim // reduction, token_dim),
        )

        # 初始化
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)

    def forward(self, batch_size):
        """
        Forward: 生成 Prompt 张量，并映射到特征空间。
        Args:
            batch_size (int): 输入数据的 batch 大小。
        Returns:
            Tensor: shape [batch_size, token_dim, 1, 1]
        """
        indices = torch.arange(self.num_virtual_tokens, device=self.embedding.weight.device)
        prompt_tokens = self.embedding(indices)  # shape: [num_virtual_tokens, token_dim]
        prompt_tokens = self.mapping(prompt_tokens)  # 映射到特征空间
        prompt_tokens = prompt_tokens.mean(dim=0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
        return prompt_tokens.expand(batch_size, -1, 1, 1)

# ResNet + Prompt 模型
class ResNetWithPrompt(nn.Module):
    def __init__(self, backbone="resnet18", num_virtual_tokens=5, num_classes=10, freeze_backbone=True):
        """
        ResNet + Prompt 模型。
        Args:
            backbone (str): ResNet 主干网络类型（支持 "resnet18" 和 "resnet50"）。
            num_virtual_tokens (int): 虚拟 token 数量。
            num_classes (int): 分类类别数量。
            freeze_backbone (bool): 是否冻结主干网络。
        """
        super(ResNetWithPrompt, self).__init__()
        # 加载预训练 ResNet
        if backbone == "resnet18":
            base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        elif backbone == "resnet50":
            base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        else:
            raise ValueError(f"Unsupported ResNet backbone: {backbone}")

        # 冻结 Backbone
        if freeze_backbone:
            for param in base_model.parameters():
                param.requires_grad = False

        self.prompt_embedding = PromptEmbedding(
            num_virtual_tokens=num_virtual_tokens, token_dim=base_model.conv1.out_channels
        )

        # 提取 ResNet 的各层
        self.conv1 = base_model.conv1
        self.bn1 = base_model.bn1
        self.relu = base_model.relu
        self.maxpool = base_model.maxpool

        self.layer1 = base_model.layer1
        self.layer2 = base_model.layer2
        self.layer3 = base_model.layer3
        self.layer4 = base_model.layer4

        self.avgpool = base_model.avgpool
        self.fc = nn.Linear(base_model.fc.in_features, num_classes)

    def forward(self, x):
        batch_size = x.size(0)

        # 生成 Prompt
        prompt = self.prompt_embedding(batch_size)

        # 第一层卷积
        x = self.conv1(x) + prompt  # 将 Prompt 添加到输入特征
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # ResNet 主干网络
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类头
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# ConvNeXt + Prompt 模型
class ConvNeXtWithPrompt(nn.Module):
    def __init__(self, backbone="convnext_tiny", num_virtual_tokens=5, num_classes=10, freeze_backbone=True):
        """
        ConvNeXt + Prompt 模型。
        Args:
            backbone (str): ConvNeXt 主干网络类型（支持 "convnext_tiny" 和 "convnext_base"）。
            num_virtual_tokens (int): 虚拟 token 数量。
            num_classes (int): 分类类别数量。
            freeze_backbone (bool): 是否冻结主干网络。
        """
        super(ConvNeXtWithPrompt, self).__init__()

        # 加载预训练 ConvNeXt
        if backbone == "convnext_tiny":
            base_model = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        elif backbone == "convnext_base":
            base_model = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported ConvNeXt backbone: {backbone}")

        # 冻结 Backbone
        if freeze_backbone:
            for param in base_model.features.parameters():
                param.requires_grad = False

        self.prompt_embedding = PromptEmbedding(
            num_virtual_tokens=num_virtual_tokens, token_dim=base_model.features[0][0].out_channels
        )

        # 提取 ConvNeXt 的各层
        self.stem = base_model.features[:1]  # 第一层
        self.features = base_model.features[1:]  # 主干网络

        # 分类头
        in_features = base_model.classifier[2].in_features
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 生成 Prompt
        prompt = self.prompt_embedding(batch_size)

        # ConvNeXt 的第一层
        x = self.stem(x) + prompt  # 将 Prompt 添加到输入特征
        x = self.features(x)

        # 分类头
        x = torch.mean(x, dim=(2, 3))  # 全局平均池化
        x = self.classifier(x)
        return x
    
# VGG + Prompt 模型
class VGGWithPrompt(nn.Module):
    def __init__(self, backbone="vgg16", num_virtual_tokens=5, num_classes=10, freeze_backbone=True):
        super(VGGWithPrompt, self).__init__()

        # 加载预训练 VGG
        if backbone == "vgg16":
            base_model = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        elif backbone == "vgg19":
            base_model = vgg19(weights=VGG19_Weights.IMAGENET1K_V1)
        else:
            raise ValueError(f"Unsupported VGG backbone: {backbone}")

        # 冻结 Backbone
        if freeze_backbone:
            for param in base_model.features.parameters():
                param.requires_grad = False

        # PromptEmbedding 的维度与特征图通道数匹配
        token_dim = base_model.features[0].out_channels
        self.prompt_embedding = PromptEmbedding(num_virtual_tokens=num_virtual_tokens, token_dim=token_dim)

        # 提取 VGG 的特征提取层
        self.features = base_model.features

        # 简化分类头
        in_features = base_model.classifier[0].in_features
        self.classifier = nn.Sequential(
            nn.Linear(in_features, num_classes)
        )

    def forward(self, x):
        batch_size = x.size(0)

        # 提取特征
        x = self.features[0](x)  # 提取第一层特征
        prompt = self.prompt_embedding(batch_size)  # 生成 Prompt
        prompt = nn.functional.interpolate(prompt, size=(x.size(2), x.size(3)), mode="bilinear", align_corners=False)

        # 添加 Prompt 到特征图
        x = x + prompt

        # 通过剩余的特征提取层
        for layer in self.features[1:]:
            x = layer(x)

        # 分类头
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


# 测试代码
if __name__ == "__main__":
    # 测试 ResNet + Prompt
    resnet_model = ResNetWithPrompt(backbone="resnet18", num_virtual_tokens=5, num_classes=10)
    x = torch.randn(4, 3, 224, 224)  # 输入数据
    output = resnet_model(x)
    print("ResNet output shape:", output.shape)

    # 测试 ConvNeXt + Prompt
    convnext_model = ConvNeXtWithPrompt(backbone="convnext_tiny", num_virtual_tokens=5, num_classes=10)
    output = convnext_model(x)
    print("ConvNeXt output shape:", output.shape)

    # 测试 VGG + Prompt
    vgg_model = VGGWithPrompt(backbone="vgg16", num_virtual_tokens=5, num_classes=10)
    x = torch.randn(4, 3, 224, 224)  # 输入数据
    output = vgg_model(x)
    print("VGG output shape:", output.shape)
