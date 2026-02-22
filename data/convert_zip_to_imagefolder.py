import os
import random
import shutil
import zipfile

def extract_zip(zip_path, extract_to):
    """ 解压 ZIP 文件 """
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print(f"解压完成: {zip_path} -> {extract_to}")

def create_imagefolder_structure(source_dir, target_dir, train_ratio=0.8):
    """
    将数据集转换为符合 ImageFolder 格式的结构，并按比例划分训练和测试集
    :param source_dir: 解压后的数据集路径
    :param target_dir: 目标路径
    :param train_ratio: 训练集的比例 (0.8 表示 80%)
    """
    train_dir = os.path.join(target_dir, "train")
    test_dir = os.path.join(target_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)

    # 遍历类别文件夹
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue  # 跳过非文件夹
        
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # 获取所有图像文件
        files = [f for f in os.listdir(class_path) if os.path.isfile(os.path.join(class_path, f))]
        random.shuffle(files)

        # 按比例划分训练集和测试集
        train_count = int(len(files) * train_ratio)
        train_files = files[:train_count]
        test_files = files[train_count:]

        # 复制文件到目标文件夹
        for file_name in train_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(train_dir, class_name, file_name))
        for file_name in test_files:
            shutil.copy(os.path.join(class_path, file_name), os.path.join(test_dir, class_name, file_name))

        print(f"类别 {class_name}: 训练集 {len(train_files)} 张，测试集 {len(test_files)} 张")

    print(f"数据集已成功划分并转换为 ImageFolder 格式，存储在 {target_dir}")

# 设置路径
zip_path = "/mnt/data/lsy/ZZQ/ISIC_2019.zip"
extract_path = "/mnt/data/lsy/ZZQ/ISIC_2019_unzipped"
target_path = "/mnt/data/lsy/ZZQ/ISIC_2019_full"

# 解压 ZIP 文件
extract_zip(zip_path, extract_path)

# 调用函数进行数据集划分
create_imagefolder_structure(extract_path, target_path)
