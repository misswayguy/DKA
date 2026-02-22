import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import sys
sys.path.append(".")
from peft_model.adapter import ResNetWithAdapter, ViTWithAdapter, SwinWithAdapter, ConvNeXtWithAdapter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略损坏的图片

import time

# 尝试导入FLOPs计算库
try:
    from fvcore.nn import FlopCountAnalysis, parameter_count
    HAS_FVCORE = True
except ImportError:
    try:
        from thop import profile, clever_format
        HAS_THOP = True
    except ImportError:
        HAS_FVCORE = False
        HAS_THOP = False
        print("Warning: Neither fvcore nor thop is installed. FLOPs calculation will be skipped.")

# 设置随机种子
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# 数据加载函数
def prepare_custom_data(data_path, preprocess):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
    test_dataset = datasets.ImageFolder(root=test_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4
    )

    class_names = train_dataset.classes
    return {"train": train_loader, "test": test_loader}, class_names

# 计算可训练参数
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# 获取 Adapter 和 Head 层的参数
def get_adapter_and_head_params(network):
    adapter_params = []
    head_params = []

    for name, param in network.named_parameters():
        if "adapter" in name:
            adapter_params.append(param)  # Adapter 层参数
        elif "head" in name or "classifier" in name:
            head_params.append(param)  # 分类层参数

    return adapter_params, head_params

# 计算FLOPs的函数
def calculate_flops(model, input_size=(1, 3, 224, 224), device="cuda"):
    """
    计算模型的FLOPs和参数数量
    """
    model.eval()
    
    # 创建随机输入
    dummy_input = torch.randn(input_size).to(device)
    
    if HAS_FVCORE:
        # 使用fvcore计算FLOPs
        flops = FlopCountAnalysis(model, dummy_input)
        total_flops = flops.total()
        params = parameter_count(model)['']
        
        # 打印每个模块的FLOPs（可选）
        print("\n=== FLOPs by module (fvcore) ===")
        for module_name, module_flops in flops.by_module().items():
            if module_flops > 0:  # 只显示有FLOPs的模块
                print(f"{module_name}: {module_flops / 1e9:.4f} GFLOPs")
                
    elif HAS_THOP:
        # 使用thop计算FLOPs
        flops, params = profile(model, inputs=(dummy_input,), verbose=False)
        total_flops = flops
        print("\n=== FLOPs (thop) ===")
        
    else:
        print("No FLOPs calculation library available.")
        return 0, 0
    
    print(f"\n📊 FLOPs Analysis:")
    print(f"Total FLOPs: {total_flops / 1e9:.4f} GFLOPs")
    print(f"Total Parameters: {params / 1e6:.4f} M")
    print(f"Trainable Parameters: {count_trainable_parameters(model) / 1e6:.4f} M")
    
    return total_flops, params

# 比较不同模型的FLOPs
def compare_flops(model_dict, input_size=(1, 3, 224, 224), device="cuda"):
    """
    比较不同模型的FLOPs
    """
    print("\n" + "="*60)
    print("🔄 COMPARING FLOPs ACROSS DIFFERENT MODELS")
    print("="*60)
    
    results = {}
    
    for model_name, model in model_dict.items():
        print(f"\n--- Calculating FLOPs for {model_name} ---")
        model = model.to(device)
        flops, params = calculate_flops(model, input_size, device)
        results[model_name] = {
            'flops_g': flops / 1e9,
            'params_m': params / 1e6,
            'trainable_params_m': count_trainable_parameters(model) / 1e6
        }
    
    # 打印比较结果
    print("\n" + "="*80)
    print("📈 FLOPs COMPARISON RESULTS")
    print("="*80)
    print(f"{'Model':<25} {'FLOPs (G)':<12} {'Params (M)':<12} {'Trainable Params (M)':<20}")
    print("-"*80)
    
    for model_name, metrics in results.items():
        print(f"{model_name:<25} {metrics['flops_g']:<12.4f} {metrics['params_m']:<12.4f} {metrics['trainable_params_m']:<20.4f}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--network',
        choices=[
            "resnet18_adapter", "vit_b16_adapter", "convnext_tiny_adapter",
            "vit_b16_adapter_low", "vit_b16_adapter_mid", "vit_b16_adapter_high",
            "swin_base_adapter", "swin_tiny_adapter"
        ],
        required=True
    )
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset")
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)      # head 学习率
    parser.add_argument('--mask_lr', type=float, default=1e-3) # adapter 学习率
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda:0")
    parser.add_argument('--compare_flops', action='store_true', help='Compare FLOPs of all models')
    args = parser.parse_args()

    # 设置设备
    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(args.seed)

    # 预处理
    preprocess = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    print(f"Loading dataset from {args.dataset}")
    loaders, class_names = prepare_custom_data(args.dataset, preprocess)
    num_classes = len(class_names)
    print("Classes:", class_names)

    # 加载模型
    model_dict = {
        "resnet18_adapter": ResNetWithAdapter(
            "resnet18", pretrained=True, reduction=64,
            freeze_backbone=True, num_classes=num_classes
        ),
        "vit_b16_adapter": ViTWithAdapter(
            "vit_b16", pretrained=True, middle_dim=16,
            freeze_backbone=True, num_classes=num_classes,
            selected_layers=list(range(0, 12))
        ),
        "vit_b16_adapter_low": ViTWithAdapter(
            "vit_b16", pretrained=True, middle_dim=10,
            freeze_backbone=True, num_classes=num_classes,
            selected_layers=list(range(0, 4))
        ),
        "vit_b16_adapter_mid": ViTWithAdapter(
            "vit_b16", pretrained=True, middle_dim=10,
            freeze_backbone=True, num_classes=num_classes,
            selected_layers=list(range(4, 8))
        ),
        "vit_b16_adapter_high": ViTWithAdapter(
            "vit_b16", pretrained=True, middle_dim=10,
            freeze_backbone=True, num_classes=num_classes,
            selected_layers=list(range(8, 12))
        ),
        "swin_tiny_adapter": SwinWithAdapter(
            "swin_t", pretrained=True, middle_dim=10,
            freeze_backbone=True, num_classes=num_classes
        ),
        "swin_base_adapter": SwinWithAdapter(
            "swin_b", pretrained=True, middle_dim=10,
            freeze_backbone=True, num_classes=num_classes
        ),
        "convnext_tiny_adapter": ConvNeXtWithAdapter(
            "convnext_tiny", pretrained=True, middle_dim=10,
            freeze_backbone=True, num_classes=num_classes
        ),
    }

    # 如果只需要比较FLOPs
    if args.compare_flops:
        compare_flops(model_dict, device=device)
        exit(0)

    # 选择特定网络
    network = model_dict[args.network].to(device)

    # 在训练前先计算FLOPs
    print(f"\n🧮 Calculating FLOPs for {args.network}...")
    calculate_flops(network, device=device)

    # 获取 Adapter 和 Head 层参数
    adapter_params, head_params = get_adapter_and_head_params(network)

    print(f"Total trainable parameters: {count_trainable_parameters(network)}")

    optimizer = torch.optim.Adam(head_params, lr=args.lr)
    mask_optimizer = torch.optim.Adam(adapter_params, lr=args.mask_lr)
    scaler = GradScaler()

    best_acc = 0.0

    # ----------------- 训练 + 测试循环 -----------------
    for epoch in range(args.epoch):
        # -------- 训练阶段 --------
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0.0
        pbar = tqdm(loaders['train'], desc=f"Epoch {epoch} Training", ncols=100)

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            mask_optimizer.zero_grad()

            with autocast():
                logits = network(x)
                loss = F.cross_entropy(logits, y)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(head_params, max_norm=1.0)
            torch.nn.utils.clip_grad_norm_(adapter_params, max_norm=1.0)
            scaler.step(optimizer)
            scaler.step(mask_optimizer)
            scaler.update()

            total_num += y.size(0)
            true_num += logits.argmax(1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)
            pbar.set_postfix_str(
                f"Acc: {100.0 * true_num / total_num:.2f}%, "
                f"Loss: {loss_sum / total_num:.4f}"
            )

        train_acc = true_num / total_num
        print(f"Epoch {epoch+1} - Train Loss: {loss_sum/total_num:.4f}, "
              f"Train Acc: {train_acc*100:.2f}%")

        # -------- 测试阶段 --------
        network.eval()
        print("Profiling inference latency and memory usage...")

        test_iter = iter(loaders["test"])
        x_test, y_test = next(test_iter)
        x_test = x_test.to(device)

        # 预热
        with torch.no_grad():
            for _ in range(10):
                _ = network(x_test)

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        with torch.no_grad():
            _ = network(x_test)
        torch.cuda.synchronize()
        end_time = time.time()

        latency = (end_time - start_time) * 1000.0  # ms
        peak_memory = torch.cuda.max_memory_allocated(device=device) / (1024 ** 2)

        print(f"Inference latency: {latency:.2f} ms | "
              f"Peak memory: {peak_memory:.2f} MB")

        # -------- 整个测试集计算 Accuracy --------
        total_num, true_num = 0, 0
        with torch.no_grad():
            for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                preds = outputs.argmax(1)

                total_num += y.size(0)
                true_num += preds.eq(y).sum().item()

        test_acc = true_num / total_num
        print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(network.state_dict(), "best_adapter_model.pth")
            print(f"*** New best model saved (Acc = {best_acc*100:.2f}%) ***")

    print(f"Training complete. Best Accuracy: {best_acc*100:.2f}%")