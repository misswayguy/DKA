# radial_psd_fiftyfive.py
import os, sys
import torch
import torch.nn as nn
import numpy as np

# 1) 把包含 fifty_five.py 的目录加到 sys.path
#    你的文件在: /home/lusiyuan/ZZQ/prompt/LKA/vit/fifty_five.py
sys.path.append("/home/lusiyuan/ZZQ/prompt/LKA/vit")

# 2) 从 fifty_five.py 导入构建函数
#    你截图里是: from fifty_five import vit_base_patch16_224_in21k as create_model
from fifty_five import vit_base_patch16_224_in21k as create_model

# --------- 频率分析函数（Nyquist 归一化）---------
def radial_psd(conv_weight, bins=128, remove_dc=True, eps=1e-12):
    k = conv_weight.shape[-1]
    N = 4 * k
    device = conv_weight.device
    dtype = conv_weight.dtype

    F = torch.fft.fft2(conv_weight, s=(N, N))
    F = torch.fft.fftshift(F, dim=(-2, -1))
    P = (F.real**2 + F.imag**2).mean(0)

    yy, xx = torch.meshgrid(
        torch.arange(N, device=device), torch.arange(N, device=device), indexing='ij'
    )
    cy = N // 2; cx = N // 2
    r_pix = torch.sqrt((yy - cy)**2 + (xx - cx)**2)
    r_norm = torch.clamp(r_pix / (N/2), 0, 1)

    if remove_dc:
        P[cy, cx] = 0

    ridx = torch.clamp((r_norm * (bins - 1)).long(), 0, bins - 1)
    psd_1d = torch.zeros(bins, device=device, dtype=dtype).scatter_add_(0, ridx.flatten(), P.flatten())
    cnts   = torch.zeros(bins, device=device, dtype=dtype).scatter_add_(0, ridx.flatten(),
                                                                        torch.ones_like(P, dtype=dtype).flatten())
    psd_1d = psd_1d / torch.clamp(cnts, min=1)

    total = torch.clamp(psd_1d.sum(), min=eps)
    cumE = psd_1d.cumsum(0) / total
    freqs = torch.linspace(0, 1, bins, device=device, dtype=dtype)

    f_c = (freqs * psd_1d).sum() / total
    f90_idx = torch.searchsorted(cumE, torch.tensor(0.9, device=device, dtype=dtype)).item()
    f90_idx = min(f90_idx, bins - 1)
    f90 = freqs[f90_idx].item()
    return psd_1d, cumE, f_c.item(), f90

def effective_kernel_from_conv(m: nn.Conv2d):
    W = m.weight.detach()
    if m.groups == m.in_channels == m.out_channels:
        return W.squeeze(1)                   # depthwise -> (C,k,k)
    else:
        return torch.linalg.norm(W, dim=1)    # 聚合输入通道 -> (C_out,k,k)

# --------- 主流程 ---------
if __name__ == "__main__":
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # A) 构建模型（按你训练时的参数）
    #   create_model(...) 的签名以你的文件为准；常见的是 num_classes/has_logits 等
    num_classes = 3   # BUSI 常见 3 类；不确定也没关系，后面 strict=False 会放行 head
    model = create_model(num_classes=num_classes)  # 如果需要 has_logits=False，就加上
    model.to(device).eval()

    # B) 加载权重
    ckpt_path = "/mnt/data/lsy/ZZQ/fifty_five_weights_vit_base_busi_80.pth"
    print(f"Loading checkpoint from: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    try:
        model.load_state_dict(state, strict=True)
    except Exception as e:
        print(f"[Info] strict=True 失败：{e}\n=> 使用 strict=False 继续加载")
        model.load_state_dict(state, strict=False)

    # C) 查找 5×5 / 51×51 卷积
    targets_5, targets_51 = [], []
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (5,5):
                targets_5.append((name, m))
            elif m.kernel_size == (51,51):
                targets_51.append((name, m))

    print(f"找到 {len(targets_5)} 个 5×5 卷积，{len(targets_51)} 个 51×51 卷积")

    # D) 计算 f_c / f90
    def analyze(m):
        W_eff = effective_kernel_from_conv(m).to(device)
        _, _, f_c, f90 = radial_psd(W_eff, bins=128, remove_dc=True)
        return f_c, f90

    vals_5  = [analyze(m) for _, m in targets_5]
    vals_51 = [analyze(m) for _, m in targets_51]

    def mean_std(arr, idx):
        if not arr: return None, None
        xs = np.array([x[idx] for x in arr], dtype=float)
        return xs.mean(), xs.std()

    if vals_5:
        mu_c, sd_c = mean_std(vals_5, 0)
        mu_90, sd_90 = mean_std(vals_5, 1)
        print(f"[5x5]  f_c={mu_c:.3f}±{sd_c:.3f}  f90={mu_90:.3f}±{sd_90:.3f}")
    if vals_51:
        mu_c, sd_c = mean_std(vals_51, 0)
        mu_90, sd_90 = mean_std(vals_51, 1)
        print(f"[51x51] f_c={mu_c:.3f}±{sd_c:.3f}  f90={mu_90:.3f}±{sd_90:.3f}")
