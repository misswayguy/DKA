import os
import random
import shutil
from PIL import Image

def clean_invalid_images(source_dir):
    """
    遍历所有数据，删除不可以读取的文件
    :param source_dir: 原始数据集路径
    """
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue
        
        for file_name in os.listdir(class_path):
            file_path = os.path.join(class_path, file_name)
            try:
                with Image.open(file_path) as img:
                    img.verify()  # 验证图片是否有效
            except (IOError, SyntaxError):
                print(f"无效文件已删除: {file_path}")
                os.remove(file_path)

def create_multiple_datasets(source_dir, target_base_dir, train_ratios, test_ratio=0.2):
    """
    创建多个完整数据集，每个数据集有不同的训练集和固定的测试集
    :param source_dir: 原始数据集路径
    :param target_base_dir: 目标数据集的根路径
    :param train_ratios: 训练集比例列表 (如 [0.005, 0.01, 0.05, 0.1, 0.2, 0.8])
    :param test_ratio: 测试集比例
    """
    # 清洗数据
    clean_invalid_images(source_dir)

    # 遍历类别，计算测试集和训练集
    class_files = {}
    for class_name in os.listdir(source_dir):
        class_path = os.path.join(source_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        files = os.listdir(class_path)
        random.shuffle(files)
        class_files[class_name] = files

    # 创建每个比例对应的数据集
    for train_ratio in train_ratios:
        dataset_name = f"dataset_train_{int(train_ratio * 100)}"
        target_dir = os.path.join(target_base_dir, dataset_name)
        train_dir = os.path.join(target_dir, "train")
        test_dir = os.path.join(target_dir, "test")

        os.makedirs(train_dir, exist_ok=True)
        os.makedirs(test_dir, exist_ok=True)

        # 遍历每个类别
        for class_name, files in class_files.items():
            class_train_dir = os.path.join(train_dir, class_name)
            class_test_dir = os.path.join(test_dir, class_name)
            os.makedirs(class_train_dir, exist_ok=True)
            os.makedirs(class_test_dir, exist_ok=True)

            # 计算测试集和训练集数量
            total_files = len(files)
            test_count = int(total_files * test_ratio)
            train_count = int(total_files * train_ratio)

            # 划分测试集（固定 20%）
            test_files = files[:test_count]

            # 划分训练集（按比例）
            train_files = files[test_count:test_count + train_count]

            # 复制测试集文件
            for file_name in test_files:
                shutil.copy(os.path.join(source_dir, class_name, file_name), os.path.join(class_test_dir, file_name))

            # 复制训练集文件
            for file_name in train_files:
                shutil.copy(os.path.join(source_dir, class_name, file_name), os.path.join(class_train_dir, file_name))

        print(f"{dataset_name}: 训练集比例 {train_ratio*100}%, 测试集比例 {test_ratio*100}%，已创建完成")

    print(f"所有数据集已创建完成，存储路径为 {target_base_dir}")

# 设置路径
source_path = "/mnt/data/lsy/ZZQ/covid"  # 原始数据集路径
target_path = "/mnt/data/lsy/ZZQ/covid_converts"  # 输出路径
train_ratios = [0.005, 0.01, 0.05, 0.1, 0.2, 0.8]  # 不同训练集比例

# 调用函数
create_multiple_datasets(source_path, target_path, train_ratios, test_ratio=0.2)
