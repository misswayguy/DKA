import os
import random
import shutil
from collections import defaultdict

def split_train_set(data_dir, output_dir, train_ratios, num_folds=5, seed=42):
    """
    根据给定比例划分训练集，创建多个训练子集，保持val验证集不变
    
    Args:
        data_dir (str): 原始数据路径，包含 'train' 和 'val' 目录
        output_dir (str): 输出路径
        train_ratios (list of float): 训练集比例，如 [0.6, 0.4, 0.2, 0.1, 0.01, 0.005]
        num_folds (int): 交叉验证折数
        seed (int): 随机种子
    """
    random.seed(seed)
    
    # 确保路径存在
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    
    if not os.path.exists(train_dir):
        raise FileNotFoundError(f"训练集路径不存在: {train_dir}")
    if not os.path.exists(val_dir):
        raise FileNotFoundError(f"验证集路径不存在: {val_dir}")

    # 1. 首先复制完整的val到每个输出目录(保持不变)
    print("处理验证集(val)...")
    for ratio in train_ratios:
        ratio_dir = os.path.join(output_dir, f"train_{int(ratio*100)}")
        os.makedirs(os.path.join(ratio_dir, "val"), exist_ok=True)
        
        # 如果val目录尚未复制
        if not os.listdir(os.path.join(ratio_dir, "val")):
            shutil.copytree(val_dir, os.path.join(ratio_dir, "val"), dirs_exist_ok=True)

    # 2. 准备训练集划分
    print("准备训练集划分...")
    class_images = defaultdict(list)
    
    # 收集所有类别的图像路径
    for class_name in os.listdir(train_dir):
        class_path = os.path.join(train_dir, class_name)
        if os.path.isdir(class_path):
            images_dir = os.path.join(class_path, "images")
            if os.path.exists(images_dir):
                images = [os.path.join(images_dir, img) 
                         for img in os.listdir(images_dir) 
                         if img.lower().endswith(('.jpg', '.jpeg', '.png'))]
                class_images[class_name] = images

    # 3. 为每个比例创建训练子集
    for ratio in train_ratios:
        print(f"\n处理训练比例: {ratio*100}%")
        ratio_dir = os.path.join(output_dir, f"train_{int(ratio*100)}")
        
        for fold in range(1, num_folds+1):
            print(f"  创建第 {fold} 折")
            fold_dir = os.path.join(ratio_dir, f"fold_{fold}")
            os.makedirs(os.path.join(fold_dir, "train"), exist_ok=True)
            
            # 为每个类别采样图像
            for class_name, images in class_images.items():
                num_total = len(images)
                num_samples = max(1, int(num_total * ratio))  # 每类至少1张
                
                # 创建类别目录
                class_output_dir = os.path.join(fold_dir, "train", class_name, "images")
                os.makedirs(class_output_dir, exist_ok=True)
                
                # 随机采样并复制图像
                sampled_images = random.sample(images, num_samples)
                for img_path in sampled_images:
                    shutil.copy(img_path, os.path.join(class_output_dir, os.path.basename(img_path)))

            # 复制完整的val到每个fold(如果尚未存在)
            val_output_dir = os.path.join(fold_dir, "val")
            if not os.path.exists(val_output_dir):
                shutil.copytree(val_dir, val_output_dir)

    print("\n所有划分完成!")

if __name__ == "__main__":
    # 配置参数
    data_dir = "/mnt/data/lsy/ZZQ/tiny-imagenet-200/tiny-imagenet-200"  # 原始数据集路径
    output_dir = "/mnt/data/lsy/ZZQ/tiny_imagenet_crosee_validation"  # 输出路径
    train_ratios = [0.6, 0.4, 0.2, 0.1, 0.01, 0.005]  # 训练集比例
    num_folds = 5  # 交叉验证折数
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行划分
    split_train_set(data_dir, output_dir, train_ratios, num_folds)