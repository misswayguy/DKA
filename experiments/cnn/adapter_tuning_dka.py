import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from tqdm import tqdm
import argparse
import sys
sys.path.append(".")
from sklearn.metrics import f1_score, roc_auc_score
from peft_model.adapter import ResNetWithAdapter, ViTWithAdapter, SwinWithAdapter, ConvNeXtWithAdapter
from PIL import ImageFile
import torch.nn as nn

ImageFile.LOAD_TRUNCATED_IMAGES = True  # 忽略损坏的图片

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

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

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

# 计算梯度
def compute_gradients(model, x, y, head_params, adapter_params):
    # Step 1: 计算分类头部的损失
    logits = model(x)
    loss_head = F.cross_entropy(logits, y)  # 分类头部的损失
    grad_head = torch.autograd.grad(loss_head, head_params, create_graph=True)

    # Step 2: 计算 Adapter 层的损失
    loss_adapter = F.cross_entropy(logits, y)  # Adapter 层的损失
    grad_adapter = torch.autograd.grad(loss_adapter, adapter_params, create_graph=True)

    return grad_head, grad_adapter

# 计算隐式梯度
def compute_implicit_gradient(grad_head, grad_adapter, head_params, adapter_params):
    # 展平 Head 层和 Adapter 层的梯度
    head_grad_vector = torch.cat([g.view(-1) for g in grad_head], dim=0)  # 展开 Head 层的梯度
    adapter_grad_vector = torch.cat([g.view(-1) for g in grad_adapter], dim=0)  # 展开 Adapter 层的梯度

    print(f"head_grad_vector size: {head_grad_vector.size()}")
    print(f"adapter_grad_vector size: {adapter_grad_vector.size()}")

    # 如果头部和 Adapter 层的梯度大小不一致，进行线性映射
    if head_grad_vector.size(0) != adapter_grad_vector.size(0):
        print(f"Warning: Gradient size mismatch! head_grad_vector size: {head_grad_vector.size(0)}, adapter_grad_vector size: {adapter_grad_vector.size(0)}")
        
        # 通过线性变换调整大小
        linear_mapping = nn.Linear(head_grad_vector.size(0), adapter_grad_vector.size(0)).to(head_grad_vector.device)
        head_grad_mapped = linear_mapping(head_grad_vector)
        adapter_grad_vector = adapter_grad_vector.view_as(head_grad_mapped)  # 将 adapter_grad_vector 调整为 head_grad_mapped 的大小
    else:
        head_grad_mapped = head_grad_vector

    # 计算隐式梯度
    implicit_gradient = -0.01 * head_grad_mapped * adapter_grad_vector  # 简单的元素级别乘法作为隐式梯度
    return implicit_gradient

# 更新参数
def update_parameters(model, grad_head, grad_adapter, head_params, adapter_params, mask_optimizer):
    # 确保 grad_head 和 grad_adapter 是有效的
    if grad_head is None or grad_adapter is None:
        print("Warning: Gradients are None, skipping update.")
        return
    
    # 计算隐式梯度
    implicit_gradient = compute_implicit_gradient(grad_head, grad_adapter, head_params, adapter_params)

    print(f"Implicit gradient size: {implicit_gradient.size()}")  # 打印隐式梯度的大小

    # 更新 Adapter 参数
    mask_optimizer.zero_grad()
    
    # 遍历 adapter 参数并更新梯度
    for param, grad in zip(adapter_params, implicit_gradient):
        if grad is not None:  # 确保 grad 不是 None
            # 只处理非标量的梯度
            if grad.ndimension() > 0:  # 确保 grad 是一个多维张量
                if grad.size(0) != param.grad.size(0):
                    print(f"Size mismatch: grad size {grad.size(0)} != param.grad size {param.grad.size(0)}")
                param.grad = grad  # 给 Adapter 参数设置隐式梯度
            else:
                print(f"Warning: grad is a scalar, skipping update for {param}")
        else:
            print(f"Warning: grad is None, skipping update for {param}")

    mask_optimizer.step()



# 主训练循环
def train(model, device, loaders, optimizer, mask_optimizer, scaler, head_params, adapter_params, epoch):
    model.train()
    total_num, true_num, loss_sum = 0, 0, 0
    pbar = tqdm(loaders['train'], desc=f"Epoch {epoch} Training", ncols=100)

    for batch_idx, (x, y) in enumerate(pbar):
        x, y = x.to(device), y.to(device)

        # Step 1: 上层优化（优化分类头部）
        with autocast():
            logits = model(x)
            loss_head = F.cross_entropy(logits, y)  # 分类头部的损失

        scaler.scale(loss_head).backward(retain_graph=True)  # retain_graph=True 保证我们后续可以计算隐式梯度
        scaler.step(optimizer)
        scaler.update()

        # Step 2: 下层优化（优化 Adapter）
        grad_head, grad_adapter = compute_gradients(model, x, y, head_params, adapter_params)

        # 更新 Adapter 参数
        update_parameters(model, grad_head, grad_adapter, head_params, adapter_params, mask_optimizer)

        # 记录训练结果
        total_num += y.size(0)
        true_num += logits.argmax(1).eq(y).sum().item()
        loss_sum += loss_head.item() * y.size(0)
        pbar.set_postfix_str(f"Acc: {100 * true_num / total_num:.2f}%, Loss: {loss_sum / total_num:.4f}")

    train_acc = true_num / total_num
    print(f"Epoch {epoch+1} - Train Loss: {loss_sum/total_num:.4f}, Train Acc: {train_acc*100:.2f}%")


# 主函数
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=["resnet18_adapter", "vit_b16_adapter", "convnext_tiny_adapter",
                                              "vit_b16_adapter_low", "vit_b16_adapter_mid", "vit_b16_adapter_high", "swin_tiny_adapter"], required=True)
    parser.add_argument('--dataset', type=str, required=True, help="Path to dataset")
    parser.add_argument('--epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)  # Head 学习率（提升收敛速度）
    parser.add_argument('--mask_lr', type=float, default=1e-5)  # Adapter 学习率（降低以防梯度爆炸）
    parser.add_argument('--seed', type=int, default=7)
    parser.add_argument('--device', type=str, default="cuda:0")
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

    # 加载模型
    model_dict = {
        "resnet18_adapter": ResNetWithAdapter("resnet18", pretrained=True, reduction=64, freeze_backbone=True, num_classes=num_classes),
        "vit_b16_adapter": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(0, 12))),
        "vit_b16_adapter_low": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(0, 4))),
        "vit_b16_adapter_mid": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(4, 8))),
        "vit_b16_adapter_high": ViTWithAdapter("vit_b16", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(8, 12))),
        "swin_tiny_adapter": SwinWithAdapter("swin_t", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes),
        "convnext_tiny_adapter": ConvNeXtWithAdapter("convnext_tiny", pretrained=True, middle_dim=10, freeze_backbone=True, num_classes=num_classes),
    }
    network = model_dict[args.network].to(device)

    # 获取 Adapter 和 Head 层参数
    adapter_params, head_params = get_adapter_and_head_params(network)

    # 优化器
    optimizer = torch.optim.Adam(head_params, lr=args.lr)  # 分类头部优化器（优化 Head）
    mask_optimizer = torch.optim.Adam(adapter_params, lr=args.mask_lr)  # Adapter 优化器（优化 Adapter）
    scaler = GradScaler()

    scheduler = torch.optim.lr_scheduler.StepLR(mask_optimizer, step_size=10, gamma=0.1)


    best_acc = 0.0

    # 训练循环
    for epoch in range(args.epoch):
        train(network, device, loaders, optimizer, mask_optimizer, scaler, head_params, adapter_params, epoch)

        # 测试阶段
        network.eval()
        total_num, true_num = 0, 0
        with torch.no_grad():
            for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                total_num += y.size(0)
                true_num += outputs.argmax(1).eq(y).sum().item()

        test_acc = true_num / total_num
        print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(network.state_dict(), "best_adapter_model.pth")

    print(f"Training complete. Best Accuracy: {best_acc*100:.2f}%")
