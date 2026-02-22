import torch
from torch import nn
from peft_model.prompt.prompted_swin_transformer import PromptedSwinTransformer

# 定义 PromptConfig 类
class PromptConfig:
    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)

class SwinWithPrompt(nn.Module):
    def __init__(self, backbone: str, num_virtual_tokens: int, num_classes: int, freeze_backbone: bool = True, pretrained_path: str = None):
        super(SwinWithPrompt, self).__init__()

        # Swin Transformer 配置
        swin_config = {
            "swin_t": {"embed_dim": 96, "depths": [2, 2, 6, 2], "num_heads": [3, 6, 12, 24]},
            "swin_s": {"embed_dim": 96, "depths": [2, 2, 18, 2], "num_heads": [3, 6, 12, 24]},
            "swin_b": {"embed_dim": 128, "depths": [2, 2, 18, 2], "num_heads": [4, 8, 16, 32]},
            "swin_l": {"embed_dim": 192, "depths": [2, 2, 18, 2], "num_heads": [6, 12, 24, 48]}
        }

        assert backbone in swin_config, f"Unsupported Swin backbone: {backbone}"
        swin_params = swin_config[backbone]

        # 使用 PromptConfig 类实例化 prompt_config
        prompt_config = PromptConfig(
            NUM_TOKENS=num_virtual_tokens,
            DROPOUT=0.0,
            LOCATION="prepend",  # 可选位置：["prepend", "add", "add-1", "pad", "below"]
            DEEP=True,
            PROJECT=-1,
            INITIATION="random"
        )

        # 初始化 PromptedSwinTransformer
        self.model = PromptedSwinTransformer(
            prompt_config=prompt_config,
            img_size=224,
            num_classes=num_classes,
            embed_dim=swin_params["embed_dim"],
            depths=swin_params["depths"],
            num_heads=swin_params["num_heads"],
        )

        # 加载预训练权重
        if pretrained_path:
            print(f"Loading pretrained weights from {pretrained_path}")
            state_dict = torch.load(pretrained_path, map_location="cpu")
            self.model.load_state_dict(state_dict, strict=False)  # 加载权重，但忽略与 prompt 相关的参数

        # 冻结 Swin Transformer 主干网络
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                param.requires_grad = False

            # 仅解冻 Prompt 模块和分类头
            for name, param in self.model.named_parameters():
                if "prompt" in name or "head" in name:
                    param.requires_grad = True

    def forward(self, x):
        return self.model(x)
