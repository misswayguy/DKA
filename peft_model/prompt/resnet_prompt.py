import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights

# 定义 PromptEmbedding 模块（基于官方代码简化）
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
    def __init__(self, num_virtual_tokens=5, num_classes=10, freeze_backbone=True):
        """
        ResNet + Prompt 模型。
        Args:
            num_virtual_tokens (int): 虚拟 token 数量。
            num_classes (int): 分类类别数量。
            freeze_backbone (bool): 是否冻结主干网络。
        """
        super(ResNetWithPrompt, self).__init__()
        # 加载预训练 ResNet
        base_model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

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
