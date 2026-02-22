import os
import shutil
import numpy as np
from sklearn.model_selection import train_test_split

def restructure_data_balanced_fixed(root_dir, output_dir, train_ratio=0.01, test_ratio=0.2, seed=42):
    """
    重新组织数据结构为 torchvision.datasets.ImageFolder 格式。
    每个类别的训练集和测试集样本数量按照比例平均分布。

    Args:
        root_dir (str): 原始数据集路径 (类别文件夹 COVID19, NORMAL, PNEUMONIA)。
        output_dir (str): 输出路径 (包含 train/ 和 test/)。
        train_ratio (float): 训练集占比。
        test_ratio (float): 测试集占比。
        seed (int): 随机种子，保证结果可复现。
    """
    np.random.seed(seed)
    categories = os.listdir(root_dir)  # 获取所有类别名称

    train_output = os.path.join(output_dir, "train")
    test_output = os.path.join(output_dir, "test")
    os.makedirs(train_output, exist_ok=True)
    os.makedirs(test_output, exist_ok=True)

    # 统计总样本数量
    total_images = sum(len(os.listdir(os.path.join(root_dir, cat))) 
                       for cat in categories if os.path.isdir(os.path.join(root_dir, cat)))

    # 计算训练集和测试集总数量
    train_total = max(1, int(total_images * train_ratio))
    test_total = max(1, int(total_images * test_ratio))

    # 平均分配到每个类别
    train_per_class = train_total // len(categories)
    test_per_class = test_total // len(categories)

    print(f"Target: Train {train_total} images, Test {test_total} images")
    print(f"Each class: Train {train_per_class} images, Test {test_per_class} images")

    # 分配训练和测试集
    for category in categories:
        category_path = os.path.join(root_dir, category)
        if not os.path.isdir(category_path):
            continue
        images = [os.path.join(category_path, img) for img in os.listdir(category_path)
                  if img.lower().endswith(('jpg', 'jpeg', 'png'))]
        np.random.shuffle(images)

        # 分别抽取训练集和测试集
        train_images = images[:train_per_class]
        test_images = images[train_per_class:train_per_class + test_per_class]

        print(f"Processing {category}: Total {len(images)}, Train {len(train_images)}, Test {len(test_images)}")

        # 输出到 train 和 test 目录
        train_category_output = os.path.join(train_output, category)
        test_category_output = os.path.join(test_output, category)
        os.makedirs(train_category_output, exist_ok=True)
        os.makedirs(test_category_output, exist_ok=True)

        for img in train_images:
            shutil.copy(img, os.path.join(train_category_output, os.path.basename(img)))
        for img in test_images:
            shutil.copy(img, os.path.join(test_category_output, os.path.basename(img)))

    print(f"Balanced fixed data successfully restructured to {output_dir} in ImageFolder format!")

# 运行函数
root_dir = "/mnt/data/lsy/ZZQ/covid"  # 原始数据路径
output_dir = "/mnt/data/lsy/ZZQ/covid_limited"  # 输出路径
restructure_data_balanced_fixed(root_dir, output_dir, train_ratio=0.01, test_ratio=0.2, seed=42)
