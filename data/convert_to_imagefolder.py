import os
import random
import shutil

def create_imagefolder_structure(source_dir, target_dir, train_ratio=0.8):
    """
    将数据集转换为符合 ImageFolder 格式的结构，并按比例划分训练和测试集
    :param source_dir: 原始数据集路径，如 /mnt/data/lsy/ZZQ/covid
    :param target_dir: 目标路径，如 /mnt/data/lsy/ZZQ/covid_full
    :param train_ratio: 训练集的比例 (0.8 表示 80%)
    """
    # 创建 train 和 test 文件夹
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历类别文件夹
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 跳过非文件夹
        
        # 创建目标路径下的类别文件夹
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # 获取该类别下所有文件
        files = os.listdir(class_path)
        random.shuffle(files)

        # 按比例划分训练集和测试集
        train_count = int(len(files) * train_ratio)
        train_files = files[:train_count]
        test_files = files[train_count:]

        # 复制文件到训练集和测试集
        for file_name in train_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(train_dir, class_name, file_name))
        for file_name in test_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(test_dir, class_name, file_name))

        print(f"类别 {class_name}: 训练集 {len(train_files)} 张，测试集 {len(test_files)} 张")

    print(f"数据集已成功划分并转换为 ImageFolder 格式，存储在 {target_dir}")

# 设置路径
source_path = "/mnt/data/lsy/ZZQ/cell.data"
target_path = "/mnt/data/lsy/ZZQ/cell.data_full"

# 调用函数
create_imagefolder_structure(source_path, target_path)
