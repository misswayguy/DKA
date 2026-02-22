import os
import zipfile
import shutil

def process_tiny_imagenet(zip_path, extract_to):
    """
    处理Tiny ImageNet数据集：
    1. 解压zip文件
    2. 删除test目录
    3. 重组val目录结构，使其与train结构一致
    
    参数:
        zip_path: zip文件路径
        extract_to: 解压到的目标目录
    """
    # 1. 解压zip文件
    print(f"正在解压 {zip_path} 到 {extract_to}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
    print("解压完成")
    
    # 获取解压后的主目录路径（处理可能的嵌套结构）
    base_dir = os.path.join(extract_to, 'tiny-imagenet-200')
    if not os.path.exists(base_dir):
        # 检查是否有嵌套的tiny-imagenet-200目录
        nested_dir = os.path.join(extract_to, 'tiny-imagenet-200', 'tiny-imagenet-200')
        if os.path.exists(nested_dir):
            base_dir = nested_dir
    
    # 2. 删除test目录
    test_dir = os.path.join(base_dir, 'test')
    if os.path.exists(test_dir):
        print(f"正在删除 {test_dir}...")
        shutil.rmtree(test_dir)
        print("test目录已删除")
    
    # 3. 重组val目录结构
    val_dir = os.path.join(base_dir, 'val')
    val_images_dir = os.path.join(val_dir, 'images')
    val_annotations_file = os.path.join(val_dir, 'val_annotations.txt')
    
    # 验证必要文件是否存在
    if not os.path.exists(val_annotations_file):
        raise FileNotFoundError(f"val_annotations.txt文件不存在于: {val_annotations_file}")
    if not os.path.exists(val_images_dir):
        raise FileNotFoundError(f"val/images目录不存在于: {val_images_dir}")
    
    # 读取wnids.txt获取所有类别
    wnids_file = os.path.join(base_dir, 'wnids.txt')
    if not os.path.exists(wnids_file):
        raise FileNotFoundError(f"wnids.txt文件不存在于: {wnids_file}")
    
    with open(wnids_file, 'r') as f:
        wnids = [line.strip() for line in f.readlines()]
    
    # 读取val标注文件
    print("正在读取val标注文件...")
    with open(val_annotations_file, 'r') as f:
        annotations = [line.strip().split('\t') for line in f.readlines()]
    
    # 创建与train相同的目录结构
    print("正在重组val目录结构...")
    
    # 在val目录下创建所有类别子目录
    for wnid in wnids:
        class_dir = os.path.join(val_dir, wnid, 'images')
        os.makedirs(class_dir, exist_ok=True)
    
    # 移动图片到对应的类别目录
    moved_count = 0
    for img_name, wnid, *_ in annotations:  # 只需要前两个字段
        src = os.path.join(val_images_dir, img_name)
        dst = os.path.join(val_dir, wnid, 'images', img_name)
        
        if os.path.exists(src):
            shutil.move(src, dst)
            moved_count += 1
        else:
            print(f"警告: 图片 {src} 不存在")
    
    print(f"成功移动 {moved_count} 张图片到对应类别目录")
    
    # 删除空的images目录
    if os.path.exists(val_images_dir):
        try:
            os.rmdir(val_images_dir)
            print("已删除空的images目录")
        except OSError:
            print(f"警告: {val_images_dir} 目录不为空，无法删除")
    
    print("val目录重组完成")
    print("数据集处理完成!")

# 使用示例
if __name__ == "__main__":
    zip_path = '/mnt/data/lsy/ZZQ/tiny-imagenet-200.zip'
    extract_to = '/mnt/data/lsy/ZZQ/tiny-imagenet-200'
    
    # 确保目标目录存在
    os.makedirs(extract_to, exist_ok=True)
    
    try:
        process_tiny_imagenet(zip_path, extract_to)
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        print("请检查:")
        print(f"1. zip文件是否存在: {zip_path}")
        print(f"2. 解压路径是否可写: {extract_to}")
        print(f"3. 解压后的目录结构是否正确")