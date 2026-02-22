import os
import torch
from torch.nn import functional as F
from torch.cuda.amp import autocast, GradScaler
from torchvision import datasets, transforms
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import argparse
import sys
sys.path.append(".")
from sklearn.metrics import f1_score, recall_score
from peft_model.adapter import ResNetWithAdapter, ViTWithAdapter, BackboneWithAdapter, SwinWithAdapter, ConvNeXtWithAdapter  # Import adapter models

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True  # Ignore truncated images

# Set random seed for reproducibility
def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# Dataset preparation function
def prepare_custom_data(data_path, preprocess):
    train_path = os.path.join(data_path, "train")
    test_path = os.path.join(data_path, "test")

    train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
    test_dataset = datasets.ImageFolder(root=test_path, transform=preprocess)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    class_names = train_dataset.classes
    return {"train": train_loader, "test": test_loader}, class_names

# def prepare_custom_data(data_path, preprocess): #tiny imageNet
#     train_path = os.path.join(data_path, "train")
#     val_path = os.path.join(data_path, "val")  # 注意这里改为val而不是test
    
#     train_dataset = datasets.ImageFolder(root=train_path, transform=preprocess)
#     val_dataset = datasets.ImageFolder(root=val_path, transform=preprocess)  # 使用val作为测试集

#     train_loader = torch.utils.data.DataLoader(
#         train_dataset, batch_size=64, shuffle=True, num_workers=4
#     )
#     val_loader = torch.utils.data.DataLoader(
#         val_dataset, batch_size=64, shuffle=False, num_workers=4
#     )

#     class_names = train_dataset.classes
#     return {"train": train_loader, "test": val_loader}, class_names  # 返回val作为test

# Count trainable parameters
def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--network', choices=["resnet18_adapter", "vit_b16_adapter", "convnext_tiny_adapter",
                                              "vit_b16_adapter_low","vit_b16_adapter_mid", "vit_b16_adapter_high",
                                               "swin_base_adapter", "swin_tiny_adapter"], required=True)
    parser.add_argument('--dataset', type=str, required=True, help="Path to the dataset root")
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--lr', type=float, default=1e-4)
    # parser.add_argument('--device', type=str, default="cuda:0", help="Device to use for training")
    parser.add_argument('--device', type=str, default="cuda:0", help='Device to use for training (e.g., cuda:0, cuda:1, cuda:2, cuda:3, cuda:4, cuda:5)')
    args = parser.parse_args()

    device = args.device if torch.cuda.is_available() else "cpu"
    set_seed(42)

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])


    dataset_name = os.path.basename(os.path.dirname(os.path.normpath(args.dataset)))

    print(f"Loading dataset from {args.dataset}")
    loaders, class_names = prepare_custom_data(args.dataset, preprocess)
    num_classes = len(class_names)

    # Load the specified adapter-based network
    if args.network == "resnet18_adapter":
        network = ResNetWithAdapter(
            backbone="resnet18", pretrained=True, reduction=64,
            freeze_backbone=True, num_classes=num_classes
        ).to(device)
    elif args.network == "vit_b16_adapter":
        network = ViTWithAdapter(
            backbone="vit_b16", pretrained=True, 
            #reduction=0.05,
            middle_dim=87,
            freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(0, 12))
        ).to(device)
    elif args.network == "vit_b16_adapter_low":
        network = ViTWithAdapter(
            backbone="vit_b16", pretrained=True, 
            #reduction=0.25,
            middle_dim=10,
            freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(0, 4))
        ).to(device)
    elif args.network == "vit_b16_adapter_mid":
        network = ViTWithAdapter(
            backbone="vit_b16", pretrained=True, 
            #reduction=768,
            middle_dim=10,
            freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(4, 8))
        ).to(device)
    elif args.network == "vit_b16_adapter_high":
        network = ViTWithAdapter(
            backbone="vit_b16", pretrained=True, 
            #reduction=768,
            middle_dim=10,
            freeze_backbone=True, num_classes=num_classes, selected_layers=list(range(8, 12))
        ).to(device)
    elif args.network == "swin_tiny_adapter":
        network = SwinWithAdapter(
            backbone="swin_t", pretrained=True,
            middle_dim=10,
            #reduction=0.05,
            freeze_backbone=True, num_classes=num_classes
        ).to(device)
    elif args.network == "swin_base_adapter":
        network = SwinWithAdapter(
            backbone="swin_b", pretrained=True,
            middle_dim=10,
            #reduction=0.05,
            freeze_backbone=True, num_classes=num_classes
        ).to(device)
    elif args.network == "convnext_tiny_adapter":
        network = ConvNeXtWithAdapter(
            backbone="convnext_tiny", pretrained=True,
            middle_dim=10,
            #reduction=0.05,
            freeze_backbone=True, num_classes=num_classes
        ).to(device)
    else:
        raise NotImplementedError(f"{args.network} is not supported")

    print(f"Total trainable parameters: {count_trainable_parameters(network)}")

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, network.parameters()), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epoch)
    scaler = GradScaler()
    best_acc = 0.0

    # Training and validation loop
    for epoch in range(args.epoch):
        network.train()
        total_num, true_num, loss_sum = 0, 0, 0

        for x, y in tqdm(loaders["train"], desc=f"Epoch {epoch+1}/{args.epoch}", ncols=100):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            with autocast():
                outputs = network(x)
                loss = F.cross_entropy(outputs, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            total_num += y.size(0)
            true_num += outputs.argmax(1).eq(y).sum().item()
            loss_sum += loss.item() * y.size(0)

        train_acc = true_num / total_num
        print(f"Epoch {epoch+1} - Train Loss: {loss_sum/total_num:.4f}, Train Acc: {train_acc*100:.2f}%")

        network.eval()
        total_num, true_num = 0, 0

        all_preds = []
        all_labels = []


        with torch.no_grad():
            for x, y in tqdm(loaders["test"], desc="Testing", ncols=100):
                x, y = x.to(device), y.to(device)
                outputs = network(x)
                preds = outputs.argmax(1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(y.cpu().numpy())
                total_num += y.size(0)
                true_num += preds.eq(y).sum().item()

        test_acc = true_num / total_num
        # f1 = f1_score(all_labels, all_preds, average='macro')  # or 'weighted'
        # sensitivity = recall_score(all_labels, all_preds, average='macro')
        # f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        # sensitivity = recall_score(all_labels, all_preds, average='macro', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='macro')
        sensitivity = recall_score(all_labels, all_preds, average='macro') 

        # print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%")
        print(f"Epoch {epoch+1} - Test Acc: {test_acc*100:.2f}%, F1: {f1:.4f}, Sensitivity: {sensitivity:.4f}")

        dataset_base_name = dataset_name.replace('/', '_')  # 用 _ 替代路径中的 / 以适应文件名

        # 构建保存路径
        save_dir = f'/mnt/data/lsy/ZZQ/{dataset_base_name}'

        # 确保目录存在
        os.makedirs(save_dir, exist_ok=True)

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(network.state_dict(), "best_adapter_model.pth")
            torch.save(network.state_dict(), os.path.join(save_dir, f"best_model_{args.network}_{dataset_base_name}.pth"))

        scheduler.step()

    print(f"Training complete. Best Accuracy: {best_acc*100:.2f}%")
