import os
import json
from collections import OrderedDict
import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler,  Subset
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split

from .dataset_lmdb import COOPLMDBDataset
from .abide import ABIDE
from .const import GTSRB_LABEL_MAP, IMAGENETNORMALIZE

from PIL import ImageFile
from PIL import UnidentifiedImageError, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

# 自定义加载器：用于在加载图像时跳过损坏图像
# def custom_loader(path):
#     try:
#         with open(path, 'rb') as f:
#             img = Image.open(f)
#             return img.convert('RGB')
#     except UnidentifiedImageError:
#         print(f"Warning: Skipping corrupted file {path}")
#         return None


def refine_classnames(class_names):
    for i, class_name in enumerate(class_names):
        class_names[i] = class_name.lower().replace('_', ' ').replace('-', ' ')
    return class_names


def get_class_names_from_split(root):
    with open(os.path.join(root, "split.json")) as f:
        split = json.load(f)["test"]
    idx_to_class = OrderedDict(sorted({s[-2]: s[-1] for s in split}.items()))
    return list(idx_to_class.values())


# def prepare_expansive_data(dataset, data_path):
#     data_path = os.path.join(data_path, dataset)
def prepare_expansive_data(dataset, data_path):
    if not isinstance(data_path, str):
        raise ValueError(f"data_path must be a string, but got {type(data_path)}: {data_path}")

    # 如果 data_path 已经是具体路径，则不再拼接 dataset
    if not os.path.exists(data_path):
        data_path = os.path.join(data_path, dataset)

    print(f"Data path for {dataset}: {data_path}") 

    if dataset == "cifar10":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR10(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "cifar100":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.CIFAR100(root = data_path, train = True, download = True, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "gtsrb":
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "svhn":
        preprocess = transforms.Compose([
            transforms.ToTensor(),
        ])
        train_data = datasets.SVHN(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': [f'{i}' for i in range(10)],
            'mask': np.zeros((32, 32)),
        }
    elif dataset == "abide":
        preprocess = transforms.ToTensor()
        D = ABIDE(root = data_path)
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': ["non ASD", "ASD"],
            'mask': D.get_mask(),
        }

    elif dataset == "covid":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
    
        # # 加载完整数据集
        # full_data = datasets.ImageFolder(root=data_path, transform=preprocess)

        # if not hasattr(full_data, 'targets') or len(full_data.targets) == 0:
        #     raise ValueError("Dataset does not contain valid targets. Check the data directory structure.")
        
    
        #train_data, test_data = balanced_data_split_fixed(full_data, dataset, train_ratio, test_ratio)

        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        }
    elif dataset == "covid_0":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")

        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        }  
    elif dataset == "covid_1":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")

        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        }  
    elif dataset == "covid_5":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")

        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        } 
    elif dataset == "covid_10":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")

        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        } 
    elif dataset == "covid_20":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")

        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        } 
    elif dataset == "covid_80":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")

        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        } 
    elif dataset == "covid_full":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        # 使用自定义加载器加载数据集
        # train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), loader=custom_loader, transform=preprocess)
        # test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), loader=custom_loader, transform=preprocess)

        # # 过滤掉加载失败的样本
        # train_data.samples = [(path, label) for path, label in train_data.samples if custom_loader(path) is not None]
        # test_data.samples = [(path, label) for path, label in test_data.samples if custom_loader(path) is not None]

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")

        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")       
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
    
        # 配置类别名和掩码尺寸
        configs = {
            #'class_names': full_data.classes, 
            'class_names': ["COVID19", "NORMAL", "PNEUMONIA"], # ImageFolder 自动提取的类名
            'mask': np.zeros((224, 224)),      # Mask 尺寸设置
        }    

    elif dataset in ["food101", "eurosat", "sun397", "ucf101", "stanfordcars", "flowers102"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        preprocess = transforms.Compose([
            transforms.Lambda(lambda x: x.convert("RGB")),
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
        ])
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
        configs = {
            'class_names': refine_classnames(test_data.classes),
            'mask': np.zeros((128, 128)),
        }
    else:
        raise NotImplementedError(f"{dataset} not supported")
    return loaders, configs


# def prepare_additive_data(dataset, data_path, preprocess):
#     data_path = os.path.join(data_path, dataset)
def prepare_additive_data(dataset, data_path, preprocess):
    if not isinstance(data_path, str):
        raise ValueError(f"data_path must be a string, but got {type(data_path)}: {data_path}")

    #save_dir = "/mnt/data/lsy/ZZQ/covid_limited_data"

    # 如果 data_path 已经是具体路径，则不再拼接 dataset
    if not os.path.exists(data_path):
        data_path = os.path.join(data_path, dataset)

    print(f"Data path for {dataset}: {data_path}") 
    if dataset == "cifar10":
        train_data = datasets.CIFAR10(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR10(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "cifar100":
        train_data = datasets.CIFAR100(root = data_path, train = True, download = False, transform = preprocess)
        test_data = datasets.CIFAR100(root = data_path, train = False, download = False, transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "svhn":
        train_data = datasets.SVHN(root = data_path, split="train", download = False, transform = preprocess)
        test_data = datasets.SVHN(root = data_path, split="test", download = False, transform = preprocess)
        class_names = [f'{i}' for i in range(10)]
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset in ["food101", "sun397", "eurosat", "ucf101", "stanfordcars", "flowers102"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=8),
        }
    elif dataset in ["dtd", "oxfordpets"]:
        train_data = COOPLMDBDataset(root = data_path, split="train", transform = preprocess)
        test_data = COOPLMDBDataset(root = data_path, split="test", transform = preprocess)
        class_names = refine_classnames(test_data.classes)
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=8),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=8),
        }
    elif dataset == "gtsrb":
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
    elif dataset == "abide":         
        D = ABIDE(root = data_path)
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((224,224)),
            transforms.Normalize(IMAGENETNORMALIZE['mean'], IMAGENETNORMALIZE['std']),
        ])
        X_train, X_test, y_train, y_test = train_test_split(D.data, D.targets, test_size=0.1, stratify=D.targets, random_state=1)
        train_data = ABIDE(root = data_path, transform = preprocess)
        train_data.data = X_train
        train_data.targets = y_train
        test_data = ABIDE(root = data_path, transform = preprocess)
        test_data.data = X_test
        test_data.targets = y_test
        loaders = {
            'train': DataLoader(train_data, 64, shuffle = True, num_workers=2),
            'test': DataLoader(test_data, 64, shuffle = False, num_workers=2),
        }
        class_names = ["non ASD", "ASD"]
    elif dataset == "covid":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]

    elif dataset == "covid_0":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]
    elif dataset == "covid_1":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]
    elif dataset == "covid_5":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]
    elif dataset == "covid_10":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]
    elif dataset == "covid_20":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]
    elif dataset == "covid_80":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]

    elif dataset == "covid_full":
        # 数据预处理
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        
        train_data = datasets.ImageFolder(root=os.path.join(data_path, "train"), transform=preprocess)
        test_data = datasets.ImageFolder(root=os.path.join(data_path, "test"), transform=preprocess)

        print(f"Train path: {os.path.join(data_path, 'train')}")
        print(f"Test path: {os.path.join(data_path, 'test')}")


        # 检查加载的样本数
        print(f"Train samples: {len(train_data)}")
        print(f"Test samples: {len(test_data)}")

        print("Class counts in train:")
        for class_idx, class_name in enumerate(train_data.classes):
            print(f"{class_name}: {sum(1 for _, label in train_data.samples if label == class_idx)}")

        print("Class counts in test:")
        for class_idx, class_name in enumerate(test_data.classes):
            print(f"{class_name}: {sum(1 for _, label in test_data.samples if label == class_idx)}")

        
        # 数据加载器
        loaders = {
            'train': DataLoader(train_data, batch_size=64, shuffle=True, num_workers=4),
            'test': DataLoader(test_data, batch_size=64, shuffle=False, num_workers=4),
        }
        #class_names = full_data.classes
        class_names = ["COVID19", "NORMAL", "PNEUMONIA"]
    else:
        raise NotImplementedError(f"{dataset} not supported")

    return loaders, class_names


def prepare_gtsrb_fraction_data(data_path, fraction, preprocess=None):
    data_path = os.path.join(data_path, "gtsrb")
    assert 0 < fraction <= 1
    new_length = int(fraction*26640)
    indices = torch.randperm(26640)[:new_length]
    sampler = SubsetRandomSampler(indices)
    if preprocess == None:
        preprocess = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
        ])
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        configs = {
            'class_names': refine_classnames(list(GTSRB_LABEL_MAP.values())),
            'mask': np.zeros((32, 32)),
        }
        return loaders, configs
    else:
        train_data = datasets.GTSRB(root = data_path, split="train", download = True, transform = preprocess)
        test_data = datasets.GTSRB(root = data_path, split="test", download = True, transform = preprocess)
        class_names = refine_classnames(list(GTSRB_LABEL_MAP.values()))
        loaders = {
            'train': DataLoader(train_data, 128, sampler=sampler, num_workers=2),
            'test': DataLoader(test_data, 128, shuffle = False, num_workers=2),
        }
        return loaders, class_names