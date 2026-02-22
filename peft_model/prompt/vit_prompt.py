import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights

class ViTWithPrompt(nn.Module):
    def __init__(
        self,
        backbone: str = "vit_b16",  # 支持 vit_b16 或 vit_l16
        pretrained: bool = True,
        num_prompts: int = 5,  # Prompt token 数量
        prompt_dim: int = None,  # Prompt 维度 (对于 vit_b16: 768, vit_l16: 1024)
        num_classes: int = 10,  # 输出类别数量
        freeze_backbone: bool = True,  # 是否冻结主干网络
        selected_blocks: list = None  # 指定在哪些 Transformer Block 中插入 Prompt tokens
    ):
        super(ViTWithPrompt, self).__init__()

        # 加载预训练的 ViT 模型
        if backbone == "vit_b16":
            self.base_model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            prompt_dim = 768
        elif backbone == "vit_l16":
            self.base_model = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
            prompt_dim = 1024
        else:
            raise ValueError("Unsupported backbone. Use 'vit_b16' or 'vit_l16'.")

        # 冻结主干网络
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        # Prompt token
        self.num_prompts = num_prompts
        self.prompt_tokens = nn.Parameter(torch.zeros(num_prompts, prompt_dim))
        nn.init.normal_(self.prompt_tokens, std=0.02)

        # 替换分类头
        self.base_model.heads.head = nn.Linear(prompt_dim, num_classes)

        # 记录要插入 Prompt 的层
        if selected_blocks is None:
            selected_blocks = []  # 默认不插入 Prompt
        self.selected_blocks = selected_blocks

    def forward(self, x):
        B = x.shape[0]

        # Initial projection (convolutional embedding)
        x = self.base_model.conv_proj(x)  # Shape: [B, C, H, W]
        x = x.flatten(2).transpose(1, 2)  # Shape: [B, num_patches, token_dim]

        # 添加 CLS token
        cls_token = self.base_model.class_token.expand(B, -1, -1)  # 修正：从 base_model 中获取 class_token
        x = torch.cat((cls_token, x), dim=1)

        # 添加位置编码
        pos_embed = self.base_model.encoder.pos_embedding[:, : x.shape[1], :]
        x = x + pos_embed

         # **在输入层插入 Prompt token**
        input_prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_prompts, prompt_dim]
        x = torch.cat((input_prompt_tokens, x), dim=1)  # 在输入位置拼接 Prompt token

        # Forward through encoder with Prompt insertion
        for idx, layer in enumerate(self.base_model.encoder.layers):
            if idx in self.selected_blocks:  # 在指定的 Block 中插入 Prompt tokens
                prompt_tokens = self.prompt_tokens.unsqueeze(0).expand(B, -1, -1)  # [B, num_prompts, prompt_dim]
                x = torch.cat((prompt_tokens, x), dim=1)  # 插入 Prompt tokens

            x = layer(x)

        # LayerNorm before classification
        x = self.base_model.encoder.ln(x)

        # 分类层
        x = self.base_model.heads(x[:, 0])  # 使用 CLS token 进行分类
        return x

def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 测试代码
if __name__ == "__main__":
    # 示例 1：仅在第 11 层插入 Prompt tokens
    model = ViTWithPrompt(backbone="vit_b16", num_prompts=5, num_classes=100, pretrained=True, selected_blocks=[11])
    print("在第 11 层插入 Prompt tokens")
    print(model)

    # 示例 2：仅在第 6 层插入 Prompt tokens
    model = ViTWithPrompt(backbone="vit_b16", num_prompts=5, num_classes=100, pretrained=True, selected_blocks=[6])
    print("\n在第 6 层插入 Prompt tokens")
    print(model)

    # 示例 3：在第 6 到第 11 层插入 Prompt tokens
    model = ViTWithPrompt(backbone="vit_b16", num_prompts=5, num_classes=100, pretrained=True, selected_blocks=list(range(6, 12)))
    print("\n在第 6-11 层插入 Prompt tokens")
    print(model)

    # 输入一个模拟的图像
    x = torch.randn(2, 3, 224, 224)  # Batch size 2, RGB 图像，224x224
    y = model(x)
    print("\n输出形状: ", y.shape)  # 输出: torch.Size([2, 100])
