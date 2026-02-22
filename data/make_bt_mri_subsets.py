import os
import shutil
import random

# 原始数据集路径
ROOT = "/mnt/data/lsy/ZZQ/bt_mri"
TRAIN_DIR = os.path.join(ROOT, "train")
TEST_DIR = os.path.join(ROOT, "test")

# 新数据集根目录（按你习惯可以改）
OUT_ROOT = "/mnt/data/lsy/ZZQ"

# 需要生成的两个比例
splits = {
    "bt_mri_0_63": 0.0063,   # 0.63%
    "bt_mri_1_25": 0.0125,   # 1.25%
}

# 可识别的图片后缀
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and os.path.splitext(f)[1].lower() in IMG_EXTS:
            files.append(f)
    return sorted(files)


def copy_test_split(out_dataset_root):
    """把整个 test 直接复制过去"""
    src = TEST_DIR
    dst = os.path.join(out_dataset_root, "test")
    if os.path.exists(dst):
        print(f"[Info] test 已存在：{dst}，跳过复制")
        return

    print(f"[Info] 复制 test 到 {dst}")
    for cls in os.listdir(src):
        cls_src = os.path.join(src, cls)
        if not os.path.isdir(cls_src):
            continue
        cls_dst = os.path.join(dst, cls)
        os.makedirs(cls_dst, exist_ok=True)

        for fname in os.listdir(cls_src):
            src_path = os.path.join(cls_src, fname)
            if os.path.isfile(src_path):
                shutil.copy2(src_path, os.path.join(cls_dst, fname))


def make_subset(split_name, ratio):
    random.seed(42)

    out_dataset_root = os.path.join(OUT_ROOT, split_name)
    out_train_root = os.path.join(out_dataset_root, "train")
    os.makedirs(out_train_root, exist_ok=True)

    # 1) 处理 train
    print(f"\n[Split] {split_name}  (ratio = {ratio * 100:.3f}%)")
    for cls in sorted(os.listdir(TRAIN_DIR)):
        cls_src_dir = os.path.join(TRAIN_DIR, cls)
        if not os.path.isdir(cls_src_dir):
            continue

        images = list_images(cls_src_dir)
        n_total = len(images)
        if n_total == 0:
            print(f"  [Warn] 类 {cls} 没有图片，跳过")
            continue

        # 每类采样数：按比例向最近整数取整，至少 1 张
        n_pick = max(1, int(round(n_total * ratio)))
        n_pick = min(n_pick, n_total)

        picked = random.sample(images, n_pick)

        cls_dst_dir = os.path.join(out_train_root, cls)
        os.makedirs(cls_dst_dir, exist_ok=True)

        for fname in picked:
            src_path = os.path.join(cls_src_dir, fname)
            dst_path = os.path.join(cls_dst_dir, fname)
            shutil.copy2(src_path, dst_path)

        print(f"  类 {cls}: 总共 {n_total} 张 -> 采样 {n_pick} 张")

    # 2) 处理 test（整套复制）
    copy_test_split(out_dataset_root)
    print(f"[Done] 子集 {split_name} 完成，路径：{out_dataset_root}")


if __name__ == "__main__":
    for name, r in splits.items():
        make_subset(name, r)
