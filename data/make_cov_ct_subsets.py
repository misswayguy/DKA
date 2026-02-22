import os
import shutil
import random

# 原始数据集路径
ROOT = "/mnt/data/lsy/ZZQ/cov_ct"   # 下面只有 COVID / non-COVID 两个文件夹
OUT_ROOT = "/mnt/data/lsy/ZZQ"      # 新数据集输出根目录

# 测试集比例（如果你想 10% 就改成 0.1）
TEST_RATIO = 0.2

# 子数据集设置：名字 -> train 比例（相对于「train pool」，也就是去掉 test 后剩下的）
SUBSETS = {
    "cov_ct_full": 1.0,      # 剩下的全部
    "cov_ct_0_63": 0.0063,   # 剩下的的 0.63%
    "cov_ct_1_25": 0.0125,   # 剩下的的 1.25%
}

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def list_images(folder):
    files = []
    for f in os.listdir(folder):
        path = os.path.join(folder, f)
        if os.path.isfile(path) and os.path.splitext(f)[1].lower() in IMG_EXTS:
            files.append(f)
    return sorted(files)


def main():
    random.seed(42)

    # 先在内存里统一划分好 test / train_pool，保证三个子数据集的 test 完全一致
    class_to_split = {}  # cls -> dict{ "test": [fname], "train_pool": [fname] }

    for cls in sorted(os.listdir(ROOT)):
        cls_dir = os.path.join(ROOT, cls)
        if not os.path.isdir(cls_dir):
            continue

        imgs = list_images(cls_dir)
        n_total = len(imgs)
        if n_total == 0:
            print(f"[Warn] 类 {cls} 没有图片，跳过")
            continue

        n_test = max(1, int(round(n_total * TEST_RATIO)))
        n_test = min(n_test, n_total - 1) if n_total > 1 else 1  # 至少保证 train_pool 还有 1 张
        test_imgs = set(random.sample(imgs, n_test))
        train_pool = [f for f in imgs if f not in test_imgs]

        class_to_split[cls] = {
            "test": sorted(test_imgs),
            "train_pool": sorted(train_pool),
        }

        print(f"类 {cls}: 总共 {n_total} 张 -> test {n_test} 张, train_pool {len(train_pool)} 张")

    # 对每一个子数据集建立目录并复制文件
    for subset_name, ratio in SUBSETS.items():
        out_root = os.path.join(OUT_ROOT, subset_name)
        train_root = os.path.join(out_root, "train")
        test_root = os.path.join(out_root, "test")

        print(f"\n[Subset] {subset_name} (train ratio on train_pool = {ratio*100:.3f}%)")
        os.makedirs(train_root, exist_ok=True)
        os.makedirs(test_root, exist_ok=True)

        for cls, split in class_to_split.items():
            # ---- 复制 test：三份数据集都一样 ----
            cls_test_dir = os.path.join(test_root, cls)
            os.makedirs(cls_test_dir, exist_ok=True)
            src_cls_dir = os.path.join(ROOT, cls)

            for fname in split["test"]:
                src = os.path.join(src_cls_dir, fname)
                dst = os.path.join(cls_test_dir, fname)
                shutil.copy2(src, dst)

            # ---- 复制 train：按子数据集比例从 train_pool 里采样 ----
            pool = split["train_pool"]
            n_pool = len(pool)
            if n_pool == 0:
                print(f"  [Warn] 类 {cls} 没有可用的 train_pool，跳过 train")
                continue

            if ratio >= 1.0:
                n_train = n_pool
            else:
                n_train = max(1, int(round(n_pool * ratio)))
                n_train = min(n_train, n_pool)

            picked = random.sample(pool, n_train)

            cls_train_dir = os.path.join(train_root, cls)
            os.makedirs(cls_train_dir, exist_ok=True)

            for fname in picked:
                src = os.path.join(src_cls_dir, fname)
                dst = os.path.join(cls_train_dir, fname)
                shutil.copy2(src, dst)

            print(f"  类 {cls}: train_pool {n_pool} 张 -> train {n_train} 张")

        print(f"[Done] {subset_name} 完成，路径：{out_root}")


if __name__ == "__main__":
    main()
