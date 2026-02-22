import torch
import torch.nn as nn

# ------------------ 配置你要测的 feature map 尺寸 ------------------
device = "cuda" if torch.cuda.is_available() else "cpu"

B = 1               # batch size
C = 192             # 通道数：分类用 16，分割用 192（你根据 DKA 设置改）
H = 512             # 特征图高度
W = 512             # 特征图宽度
iters = 100         # 重复次数，越大越稳定

print(f"Device: {device}")
print(f"Input shape: B={B}, C={C}, H={H}, W={W}")
print(f"Iterations: {iters}")

# ------------------ 定义三个 depthwise 模块 ------------------

def make_dw_conv(C, k):
    padding = k // 2
    conv = nn.Conv2d(
        in_channels=C,
        out_channels=C,
        kernel_size=k,
        padding=padding,
        groups=C,         # depthwise 关键
        bias=False
    )
    return conv.to(device).eval()

class DW5(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw = make_dw_conv(C, 5)

    def forward(self, x):
        return self.dw(x)

class DW51(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw = make_dw_conv(C, 51)

    def forward(self, x):
        return self.dw(x)

class DW5_51(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.dw5 = make_dw_conv(C, 5)
        self.dw51 = make_dw_conv(C, 51)

    def forward(self, x):
        return self.dw5(x) + self.dw51(x)

# ------------------ 通用 benchmark 函数 ------------------

def benchmark(module, x, name, iters=100):
    module = module.to(device).eval()
    x = x.to(device)

    # 预热几次（让 cudnn / kernel 稳定下来）
    with torch.no_grad():
        for _ in range(10):
            _ = module(x)

    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start_event.record()
        for _ in range(iters):
            _ = module(x)
        end_event.record()

    torch.cuda.synchronize()

    avg_ms = start_event.elapsed_time(end_event) / iters   # 单位：ms
    max_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB

    print(f"[{name}]  avg latency: {avg_ms:.4f} ms  |  peak memory: {max_mem:.2f} MB")


def main():
    # 随机输入（模拟 DKA 里的中间特征）
    x = torch.randn(B, C, H, W, device=device)

    m5 = DW5(C)
    m51 = DW51(C)
    m5_51 = DW5_51(C)

    print("\n==== Benchmark depthwise convolutions ====\n")
    benchmark(m5, x, "5x5 only", iters=iters)
    benchmark(m51, x, "51x51 only", iters=iters)
    benchmark(m5_51, x, "5x5 + 51x51", iters=iters)


if __name__ == "__main__":
    if device == "cpu":
        print("⚠ 当前没有检测到 CUDA，建议在 GPU 上跑以获得 reviewer 关心的数字。")
    main()
