import os
import random
import shutil
from collections import defaultdict

# 设置随机种子以保证结果可复现
random.seed(42)

def create_folds(data_dir, output_dir, train_ratios, num_folds=5):
    """
    根据给定的训练比例创建多个折的数据集。

    Args:
        data_dir (str): 原始数据路径，必须包含 'train' 和 'test' 目录。
        output_dir (str): 输出路径。
        train_ratios (list of float): 训练集比例，例如 [0.005, 0.01, 0.05, 0.1]。
        num_folds (int): 创建的折数。
    """
    # 确保输出路径存在
    os.makedirs(output_dir, exist_ok=True)

    # 加载原始数据集
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    if not os.path.exists(train_dir) or not os.path.exists(test_dir):
        raise FileNotFoundError("训练集或测试集路径不存在，请检查数据路径。")

    # 遍历训练集中所有类别
    class_to_images = defaultdict(list)
    for cls_name in os.listdir(train_dir):
        cls_path = os.path.join(train_dir, cls_name)
        if os.path.isdir(cls_path):
            images = [os.path.join(cls_path, img) for img in os.listdir(cls_path) if img.lower().endswith(('jpg', 'jpeg', 'png'))]
            class_to_images[cls_name].extend(images)

    # 创建每个训练比例的数据集
    for ratio in train_ratios:
        print(f"Processing train ratio {ratio * 100:.2f}%")
        # ratio_output_dir = os.path.join(output_dir, f"train_{int(ratio * 100)}")
        ratio_output_dir = os.path.join(output_dir, f"train_{ratio:.3f}".replace(".", "_"))
        os.makedirs(ratio_output_dir, exist_ok=True)

        for fold in range(1, num_folds + 1):
            print(f"  Creating fold {fold}")
            fold_output_dir = os.path.join(ratio_output_dir, f"fold_{fold}")
            train_output_dir = os.path.join(fold_output_dir, "train")
            test_output_dir = os.path.join(fold_output_dir, "test")

            # 创建目录
            os.makedirs(train_output_dir, exist_ok=True)
            os.makedirs(test_output_dir, exist_ok=True)

            # 将测试集固定复制到每个折的测试目录
            for cls_name in os.listdir(test_dir):
                cls_test_path = os.path.join(test_dir, cls_name)
                if os.path.isdir(cls_test_path):
                    cls_output_path = os.path.join(test_output_dir, cls_name)
                    os.makedirs(cls_output_path, exist_ok=True)
                    for img in os.listdir(cls_test_path):
                        shutil.copy(os.path.join(cls_test_path, img), os.path.join(cls_output_path, img))

            # 为每个折划分训练集
            for cls_name, images in class_to_images.items():
                cls_train_output_dir = os.path.join(train_output_dir, cls_name)
                os.makedirs(cls_train_output_dir, exist_ok=True)

                num_images = len(images)
                num_train_images = max(1, int(num_images * ratio))  # 确保每个类别至少有一张图像

                # 随机采样用于训练的图像
                sampled_images = random.sample(images, num_train_images)

                for img_path in sampled_images:
                    shutil.copy(img_path, os.path.join(cls_train_output_dir, os.path.basename(img_path)))

if __name__ == "__main__":
    data_dir = "/mnt/data/lsy/ZZQ/ISIC_2019_full"  # 原始数据集路径
    output_dir = "/mnt/data/lsy/ZZQ/ISIC_2019_full_cross_validation_v2"  # 输出路径
    train_ratios = [0.001]  # 训练集比例
    num_folds = 5  # 折数

    create_folds(data_dir, output_dir, train_ratios, num_folds)
