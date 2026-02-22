import os
import tarfile
import pickle
import numpy as np
from PIL import Image
from tqdm import tqdm

# 配置路径（根据你的需求修改）
TAR_GZ_PATH = '/mnt/data/lsy/ZZQ/cifar-100-python.tar.gz'
# EXTRACT_DIR = '/mnt/data/lsy/ZZQ/cifar-100-python'

# 修正后（指向内层目录）
EXTRACT_DIR = '/mnt/data/lsy/ZZQ/cifar-100-python/cifar-100-python'  # 实际包含 train/test 的目录
SAVE_DIR = '/mnt/data/lsy/ZZQ/cifar-100-images'

# 1. 解压 .tar.gz 文件
def extract_tar_gz():
    if not os.path.exists(TAR_GZ_PATH):
        raise FileNotFoundError(f"压缩包 {TAR_GZ_PATH} 不存在！")
    
    print("解压文件中...")
    with tarfile.open(TAR_GZ_PATH, 'r:gz') as tar:
        tar.extractall(path=EXTRACT_DIR)
    print(f"解压完成！文件已保存到 {EXTRACT_DIR}")

# 2. 加载二进制数据并保存为图片
def save_as_images():
    # 确保解压后的文件存在
    train_path = os.path.join(EXTRACT_DIR, 'train')
    test_path = os.path.join(EXTRACT_DIR, 'test')
    if not os.path.exists(train_path):
        raise FileNotFoundError("解压后的训练集文件不存在！")
    
    # 加载数据
    def load_batch(file_path):
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        return batch
    
    print("加载训练集和测试集数据...")
    train_batch = load_batch(train_path)
    test_batch = load_batch(test_path) if os.path.exists(test_path) else None
    
    # 合并数据（可选）
    images = np.concatenate([train_batch[b'data'], test_batch[b'data']]) if test_batch else train_batch[b'data']
    fine_labels = train_batch[b'fine_labels'] + test_batch[b'fine_labels'] if test_batch else train_batch[b'fine_labels']
    
    # 创建保存目录（按类别分文件夹）
    os.makedirs(SAVE_DIR, exist_ok=True)
    for label in set(fine_labels):
        os.makedirs(os.path.join(SAVE_DIR, str(label)), exist_ok=True)
    
    # 转换为图片并保存
    print("保存图片中...")
    for i, (img, label) in enumerate(tqdm(zip(images, fine_labels), total=len(images))):
        img_rgb = img.reshape(3, 32, 32).transpose(1, 2, 0)  # 转换为 HWC 格式
        pil_img = Image.fromarray(img_rgb)
        pil_img.save(os.path.join(SAVE_DIR, str(label), f"{i}.png"))
    
    print(f"图片保存完成！总计 {len(images)} 张图片，路径：{SAVE_DIR}")

# 主函数
if __name__ == '__main__':
    extract_tar_gz()
    save_as_images()