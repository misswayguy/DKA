import os
import random
import shutil
from collections import defaultdict
import numpy as np

def split_cifar100_dataset(data_dir, output_dir, train_ratios=[0.6, 0.4, 0.2, 0.1, 0.01, 0.005], test_ratio=0.2, seed=42):
    """
    划分CIFAR-100数据集，创建不同比例的训练子集，保持测试集不变
    
    Args:
        data_dir (str): 原始数据路径，包含0-99类别文件夹
        output_dir (str): 输出路径
        train_ratios (list): 训练子集比例列表
        test_ratio (float): 测试集比例
        seed (int): 随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    
    # 创建输出目录结构
    os.makedirs(os.path.join(output_dir, "full"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "full", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "full", "test"), exist_ok=True)
    
    # 1. 首先划分完整训练集和测试集
    print("划分完整训练集和测试集...")
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            # 创建完整集的类别目录
            os.makedirs(os.path.join(output_dir, "full", "train", class_name), exist_ok=True)
            os.makedirs(os.path.join(output_dir, "full", "test", class_name), exist_ok=True)
            
            # 获取该类所有图片
            images = [f for f in os.listdir(class_path) if f.endswith('.png')]
            random.shuffle(images)
            
            # 划分训练测试集
            split_idx = int(len(images) * (1 - test_ratio))
            train_images = images[:split_idx]
            test_images = images[split_idx:]
            
            # 复制到完整集目录
            for img in train_images:
                shutil.copy(os.path.join(class_path, img),
                          os.path.join(output_dir, "full", "train", class_name, img))
            for img in test_images:
                shutil.copy(os.path.join(class_path, img),
                          os.path.join(output_dir, "full", "test", class_name, img))
    
    # 2. 准备训练集划分
    print("\n准备训练子集划分...")
    class_images = defaultdict(list)
    
    # 收集所有类别的训练图像路径
    full_train_dir = os.path.join(output_dir, "full", "train")
    for class_name in os.listdir(full_train_dir):
        class_path = os.path.join(full_train_dir, class_name)
        images = [os.path.join(class_path, img) for img in os.listdir(class_path)]
        class_images[class_name] = images
    
    # 3. 为每个比例创建训练子集
    for ratio in train_ratios:
        print(f"\n处理训练比例: {ratio*100}%")
        ratio_dir = os.path.join(output_dir, f"train_{int(ratio*100)}")
        os.makedirs(os.path.join(ratio_dir, "train"), exist_ok=True)
        os.makedirs(os.path.join(ratio_dir, "test"), exist_ok=True)
        
        # 复制完整的测试集
        if not os.listdir(os.path.join(ratio_dir, "test")):
            shutil.copytree(os.path.join(output_dir, "full", "test"),
                           os.path.join(ratio_dir, "test"),
                           dirs_exist_ok=True)
        
        # 为每个类别采样图像
        for class_name, images in class_images.items():
            num_total = len(images)
            num_samples = max(1, int(num_total * ratio))  # 每类至少1张
            
            # 创建类别目录
            os.makedirs(os.path.join(ratio_dir, "train", class_name), exist_ok=True)
            
            # 随机采样并复制图像
            sampled_images = random.sample(images, num_samples)
            for img_path in sampled_images:
                shutil.copy(img_path, 
                          os.path.join(ratio_dir, "train", class_name, os.path.basename(img_path)))

    print("\n所有划分完成!")
    print(f"完整数据集路径: {os.path.join(output_dir, 'full')}")
    print(f"不同比例训练集路径: {output_dir}/train_XX")

if __name__ == "__main__":
    # 配置参数
    data_dir = "/mnt/data/lsy/ZZQ/cifar-100-images"  # 原始CIFAR-100图片路径(0-99类别文件夹)
    output_dir = "/mnt/data/lsy/ZZQ/cifar-100_cross_validation"  # 输出路径
    train_ratios = [0.8, 0.6, 0.4, 0.2, 0.1, 0.01, 0.005]  # 训练集比例
    
    # 运行划分
    split_cifar100_dataset(data_dir, output_dir, train_ratios)