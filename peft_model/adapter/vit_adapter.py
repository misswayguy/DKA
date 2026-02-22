import torch
import torch.nn as nn
from torchvision.models import vit_b_16, vit_l_16, ViT_B_16_Weights, ViT_L_16_Weights
import torch.nn.functional as F
import torch.nn.init as init

# class Adapter(nn.Module):
#     def __init__(self, input_dim, middle_dim=None, reduction=0.25):
#         super(Adapter, self).__init__()

#         if input_dim * reduction < 1:   
#             raise ValueError(f"Invalid reduction value: {reduction}. "
#                              f"Middle dimension (input_dim // reduction) must be >= 1, "
#                              f"but got {input_dim // reduction}.")

#         self.adapter = nn.Sequential(
#             nn.Linear(input_dim, middle_dim, bias=False),
#             #nn.Linear(input_dim, int(input_dim * reduction), bias=False),
#             nn.ReLU(),
#             nn.Linear(middle_dim, input_dim, bias=False),
#             #nn.Linear(int(input_dim * reduction), input_dim, bias=False),
#         )

#     def forward(self, x):
#         return x + self.adapter(x)  # Residual connection

class Adapter(nn.Module):
    def __init__(self, input_dim, middle_dim=None, reduction=0.25):
        super(Adapter, self).__init__()

        # 线性降维，等效于 1x1 卷积
        self.adapter_down = nn.Linear(input_dim, middle_dim, bias=False)
        self.adapter_up = nn.Linear(middle_dim, input_dim, bias=False)

        # # 初始化为 0
        # init.zeros_(self.adapter_down.weight)  # 初始化 adapter_down 权重为0
        # init.zeros_(self.adapter_up.weight)  # 初始化 adapter_up 权重为0

         # 11×11 深度可分离卷积
        # self.conv1 = nn.Conv2d(middle_dim, middle_dim, kernel_size=11, stride=1, padding=5, groups=middle_dim, bias=False)

        # 31×31 深度可分离卷积
        # self.conv1 = nn.Conv2d(middle_dim, middle_dim, kernel_size=31, stride=1, padding=15, groups=middle_dim, bias=False)

        # 51×51 深度可分离卷积
        self.conv1 = nn.Conv2d(middle_dim, middle_dim, kernel_size=51, stride=1, padding=25, groups=middle_dim, bias=False)

        # 71×71 深度可分离卷积
        # self.conv1 = nn.Conv2d(middle_dim, middle_dim, kernel_size=71, stride=1, padding=35, groups=middle_dim, bias=False)

         # 71×71 深度可分离卷积
        # self.conv1 = nn.Conv2d(middle_dim, middle_dim, kernel_size=71, stride=1, padding=35, groups=middle_dim, bias=False)

        # 5×5 深度可分离卷积
        self.conv2 = nn.Conv2d(middle_dim,middle_dim, kernel_size=5, stride=1, padding=2, groups=middle_dim, bias=False)
        # # 3×3 深度可分离卷积
        # self.conv2 = nn.Conv2d(middle_dim, middle_dim, kernel_size=3, stride=1, padding=1, groups=middle_dim, bias=False)
        # # 7×7 深度可分离卷积
        # self.conv2 = nn.Conv2d(middle_dim, middle_dim, kernel_size=7, stride=1, padding=3, groups=middle_dim, bias=False)
        # # 9×9 深度可分离卷积
        # self.conv2 = nn.Conv2d(middle_dim, middle_dim, kernel_size=9, stride=1, padding=4, groups=middle_dim, bias=False)
        
        self.conv3 = nn.Conv2d(middle_dim,middle_dim, kernel_size=11, stride=1, padding=5, groups=middle_dim, bias=False)
        
        self.conv_local = nn.Conv2d(
            middle_dim, middle_dim,
            kernel_size=3, dilation=2, padding=2,  # padding = dilation
            stride=1, groups=middle_dim, bias=False
        )
        
        
        self.conv_global = nn.Conv2d(
            middle_dim, middle_dim,
            kernel_size=3, dilation=25, padding=25,  # padding = dilation
            stride=1, groups=middle_dim, bias=False
        )


        self.conv_global_2 = nn.Conv2d(
            middle_dim, middle_dim,
            kernel_size=11, dilation=5, padding=25,  # padding = dilation * (k - 1) // 2 = 5 * 5 = 25
            stride=1, groups=middle_dim, bias=False
        )

        self.conv_global_3 = nn.Conv2d(
            middle_dim, middle_dim,
            kernel_size=26, dilation=2, padding=25,  # (26-1)×2 = 50
            stride=1, groups=middle_dim, bias=False
        )

        self.alpha = nn.Parameter(torch.zeros(1))  # 对应大核
        self.beta  = nn.Parameter(torch.zeros(1))  # 对应小核
        
        # 激活函数 + Dropout
        self.act = F.gelu
        self.dropout = nn.Dropout(0.1)


# # class Adapter(nn.Module):
# #     def __init__(self, input_dim, middle_dim=None, reduction=0.25):
# #         super(Adapter, self).__init__()

# #         self.adapter_down = nn.Linear(input_dim, middle_dim, bias=False)
# #         self.adapter_up   = nn.Linear(middle_dim, input_dim, bias=False)

# #         # 大核 & 小核（你现在的 DKA 分支）
# #         self.conv_big  = nn.Conv2d(middle_dim, middle_dim, kernel_size=51, stride=1, padding=25, groups=middle_dim, bias=False)
# #         self.conv_small = nn.Conv2d(middle_dim, middle_dim, kernel_size=5,  stride=1, padding=2,  groups=middle_dim, bias=False)

# #         # ⭐ 关键：可学习的两个标量，用来控制两条分支的权重
# #         # 初始设为 0，softmax 后就是 0.5 / 0.5，相当于一开始平均。
# #         self.alpha = nn.Parameter(torch.zeros(1))  # 对应大核
# #         self.beta  = nn.Parameter(torch.zeros(1))  # 对应小核

# #         self.act = F.gelu
# #         self.dropout = nn.Dropout(0.1)


#     def forward(self, x):
#         B, N, C = x.shape  # (batch, seq_len, embed_dim)

#         # Step 1: 线性降维  
#         x_down = self.adapter_down(x)  # (B, N, reduced_dim)
#         x_down = self.act(x_down)

#         # Step 2: 变形适配 Conv2D
#         H = W = int((N - 1) ** 0.5)  # 假设 cls_token 之外的是正方形 patch grid
#         x_patch = x_down[:, 1:].reshape(B, H, W, -1).permute(0, 3, 1, 2)  # 变为 (B, reduced_dim, H, W)

#         # Step 3: 计算并联卷积
#         conv1_out = self.conv1(x_patch)  # 31×31
#         conv2_out = self.conv2(x_patch)  # 5×5
#         # conv3_out = self.conv3(x_patch)  # 5×5
#         x_patch = conv1_out + conv2_out  # 逐元素相加

#         # x_patch = conv1_out
#         # x_patch = self.conv2(x_patch)  # 31×31

#         # Step 4: 变回 Transformer 格式
#         x_patch = x_patch.permute(0, 2, 3, 1).reshape(B, H * W, -1)  # (B, N-1, reduced_dim)

#         # Step 5: 处理 cls_token
#         x_cls = x_down[:, :1]  # (B, 1, reduced_dim)

#         # Step 6: 连接
#         x_down = torch.cat([x_cls, x_patch], dim=1)  # (B, N, reduced_dim)
#         x_down = self.act(x_down)
#         x_down = self.dropout(x_down)

#         # Step 7: 线性升维 + 残差连接
#         x_up = self.adapter_up(x_down)  # (B, N, embed_dim)

#         return x + x_up  # 残差连接，保持维度不变



class ViTWithAdapter(nn.Module):
    def __init__(self, backbone="vit_b16", pretrained=True, reduction=16, middle_dim=None, freeze_backbone=True, num_classes=10, selected_layers=None):
        super(ViTWithAdapter, self).__init__()

        # Load pre-trained ViT model
        if backbone == "vit_b16":
            self.vit = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dim = 768
        elif backbone == "vit_l16":
            self.vit = vit_l_16(weights=ViT_L_16_Weights.IMAGENET1K_V1 if pretrained else None)
            hidden_dim = 1024
        else:
            raise ValueError("Unsupported backbone. Use 'vit_b16' or 'vit_l16'.")

        # Freeze backbone if needed
        if freeze_backbone:
            for param in self.vit.parameters():
                param.requires_grad = False

        # 默认只在最后一层添加 Adapter
        if selected_layers is None:
            selected_layers = [11]  # 默认只在第 11 层

        # Add Adapter to the selected Transformer Blocks
        for idx, block in enumerate(self.vit.encoder.layers):
            if idx in selected_layers:
                block.adapter = Adapter(hidden_dim, middle_dim=middle_dim, reduction=reduction)
                original_forward = block.forward

                def forward_with_adapter(x, original_forward=original_forward, adapter=block.adapter):
                    x = original_forward(x)  # Original forward pass
                    x = adapter(x)  # Pass through Adapter
                    return x

                block.forward = forward_with_adapter

        # Replace the classification head
        self.vit.heads.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        x = self.vit(x)
        return x


# Test code
if __name__ == "__main__":
    # Test ViT-B/16 + Adapter
    vit_b_model = ViTWithAdapter(backbone="vit_b16", num_classes=10)
    x = torch.randn(4, 3, 224, 224)  # Input data
    output = vit_b_model(x)
    print("ViT-B/16 output shape:", output.shape)

    # Test ViT-L/16 + Adapter
    vit_l_model = ViTWithAdapter(backbone="vit_l16", num_classes=10)
    output = vit_l_model(x)
    print("ViT-L/16 output shape:", output.shape)
