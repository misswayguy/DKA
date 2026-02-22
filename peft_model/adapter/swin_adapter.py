import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.swin_transformer import (
    swin_t, swin_s, swin_b,
    Swin_T_Weights, Swin_S_Weights, Swin_B_Weights
)

class Adapter(nn.Module):
    def __init__(self, input_dim, middle_dim=10, reduction=None):
        super(Adapter, self).__init__()

        self.middle_dim = middle_dim if middle_dim else max(1, int(input_dim * reduction))  # 确保 middle_dim >= 1
        self.adapter_down = nn.Linear(input_dim, self.middle_dim, bias=False)
        self.adapter_up = nn.Linear(self.middle_dim, input_dim, bias=False)

        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, H, W, C = x.shape  # 保持 Swin 格式 (B, H, W, C)
        x = x.view(B, H * W, C)  # 变成 (B, N, C)

        # **自动调整 `adapter_down` 的 `input_dim`**
        if x.shape[-1] != self.adapter_down.in_features:
            self.adapter_down = nn.Linear(x.shape[-1], self.middle_dim, bias=False).to(x.device)
            self.adapter_up = nn.Linear(self.middle_dim, x.shape[-1], bias=False).to(x.device)

        x_down = self.adapter_down(x)  # 线性降维
        x_down = self.act(x_down)

        x_down = self.dropout(x_down)
        x_up = self.adapter_up(x_down)  # 线性升维

        return x.view(B, H, W, C) + x_up.view(B, H, W, C)  # **残差连接，形状匹配**


# class Adapter(nn.Module):
#     def __init__(self, input_dim, middle_dim=None, reduction=0.25):
#         super(Adapter, self).__init__()

#         self.middle_dim = middle_dim if middle_dim else int(input_dim * reduction)
#         self.adapter_down = nn.Linear(input_dim, self.middle_dim, bias=False)
#         self.adapter_up = nn.Linear(self.middle_dim, input_dim, bias=False)

#         self.conv1 = nn.Conv2d(self.middle_dim, self.middle_dim, kernel_size=31, stride=1, padding=15, groups=self.middle_dim, bias=False)
#         self.conv2 = nn.Conv2d(self.middle_dim, self.middle_dim, kernel_size=5, stride=1, padding=2, groups=self.middle_dim, bias=False)

#         self.act = F.gelu
#         self.dropout = nn.Dropout(0.1)

#     def forward(self, x):
#         B, H, W, C = x.shape
#         x = x.view(B, H * W, C)

#         if x.shape[-1] != self.adapter_down.in_features:
#             self.adapter_down = nn.Linear(x.shape[-1], self.middle_dim, bias=False).to(x.device)
#             self.adapter_up = nn.Linear(self.middle_dim, x.shape[-1], bias=False).to(x.device)

#         x_down = self.adapter_down(x)
#         x_down = self.act(x_down)

#         x_patch = x_down.reshape(B, H, W, -1).permute(0, 3, 1, 2)
#         conv1_out = self.conv1(x_patch)
#         conv2_out = self.conv2(x_patch)
#         x_patch = conv1_out + conv2_out

#         x_patch = x_patch.permute(0, 2, 3, 1)
#         x_down = self.act(x_patch)
#         x_down = self.dropout(x_down)

#         x_up = self.adapter_up(x_down)
#         return x.view(B, H, W, C) + x_up


class SwinWithAdapter(nn.Module):
    def __init__(self, backbone='swin_t', pretrained=True, reduction=16, middle_dim=None, freeze_backbone=True, num_classes=10, selected_layers=None):
        super(SwinWithAdapter, self).__init__()

        if backbone == 'swin_t':
            self.swin = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dims = [96, 192, 384, 768]
        elif backbone == 'swin_s':
            self.swin = swin_s(weights=Swin_S_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dims = [96, 192, 384, 768]
        elif backbone == 'swin_b':
            self.swin = swin_b(weights=Swin_B_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dims = [128, 256, 512, 1024]
        else:
            raise ValueError("Unsupported Swin Transformer backbone.")

        if freeze_backbone:
            for param in self.swin.parameters():
                param.requires_grad = False

        # if selected_layers is None:
        #     # selected_layers = {0: [1, 3], 1: [1], 2: [], 3: []}  # 选择特定层
        #     selected_layers = {0: [0, 1, 2, 3], 1: [0, 1, 2, 3], 2: [0, 1, 2, 3], 3: [0, 1, 2, 3]}

        swin_config = {
            'swin_t': [2, 2, 6, 2],
            'swin_s': [2, 2, 18, 2],
            'swin_b': [2, 2, 18, 2]
        }
        if selected_layers is None:
            selected_layers = {
                stage_idx: list(range(num_blocks))
                for stage_idx, num_blocks in enumerate(swin_config[backbone])
            }


        for stage_idx, stage in enumerate(self.swin.features):
            if isinstance(stage, nn.Sequential):
                for block_idx, block in enumerate(stage):
                    if stage_idx in selected_layers and block_idx in selected_layers[stage_idx]:
                        input_dim = hidden_dims[stage_idx]
                        block.adapter = Adapter(input_dim, middle_dim=middle_dim, reduction=reduction)
                        original_forward = block.forward

                        def forward_with_adapter(x, original_forward=original_forward, adapter=block.adapter):
                            x = original_forward(x)
                            x = adapter(x)
                            return x

                        block.forward = forward_with_adapter

        self.swin.head = nn.Linear(hidden_dims[-1], num_classes)

    def forward(self, x):
        return self.swin(x)



if __name__ == "__main__":
    model = SwinWithAdapter(backbone="swin_t", num_classes=10)
    x = torch.randn(4, 3, 224, 224)
    output = model(x)
    print("Swin-T output shape:", output.shape)

    model = SwinWithAdapter(backbone="swin_b", num_classes=10)
    output = model(x)
    print("Swin-B output shape:", output.shape)


