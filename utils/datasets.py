from torchvision import datasets, transforms
from torch.utils.data import Dataset
import os
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import torchvision
from PIL import Image
from torchvision.datasets import VisionDataset
from sklearn.model_selection import train_test_split
import numpy as np
import torch

from ucimlrepo import fetch_ucirepo 



feature_sizes = []


class ImageNet12(Dataset):
    def __init__(self, root="./dataset", train=True, transform=None, **kwargs):
        super().__init__()
        self.train = train
        self.transform = transform
        
        root = os.path.join(root, "imagenet12")

        if train:
            fold = '/home/shunjie/codes/tifs/data/imagenet12/train'
        else:
            fold = '/home/shunjie/codes/tifs/data/imagenet12/val'
        self.dataset = torchvision.datasets.ImageFolder(root=fold, transform=self.transform)

    def __getitem__(self, idx):
        img, label = self.dataset[idx][0], self.dataset[idx][1]
        return img, label, idx

    def __len__(self):
        return len(self.dataset)

class CINIC10(VisionDataset):
    def __init__(self, root="./dataset", train=True, transform=None, **kwargs):
        super(CINIC10, self).__init__(root, transform=transform)
        self.train = train
        root = os.path.join(root, "CINIC10")

        if not os.path.exists(root):
            raise ValueError("You should download and unzip the CINIC10 dataset to {} first! Download: https://github.com/BayesWatch/cinic-10".format(root))

        if train:
            fold = '/train'
        else:
            fold = '/test'
        image = torchvision.datasets.ImageFolder(root=root + fold, transform=transform)
        self.data = image.imgs
        self.transform = transform

    def __getitem__(self, idx):
        path, label = self.data[idx]
        img = Image.open(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label, idx

    def __len__(self):
        return len(self.data)


class Criteo(Dataset):
    '''
    To load Criteo dataset.
    '''
    def __init__(self, root="./dataset", train=True, **kwargs):
        self.train = train
        root = os.path.join(root, "criteo")

        if not os.path.exists(root):
            raise ValueError("You should download and unzip the Criteo dataset to {} first! Download: https://go.criteo.net/criteo-research-kaggle-display-advertising-challenge-dataset.tar.gz".format(root))
        
        # sample data
        file_out = "train_sampled.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            file_in = "train.txt"
            self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', nrows=70000, index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                if idx > 0 and idx <= 13:
                    self.csv_data[col] = self.csv_data[col].fillna(0,)
                elif idx >= 14:
                    self.csv_data[col] = self.csv_data[col].fillna('-1',)

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset sampling completed.")
        
        # process data
        file_out = "train_processed.txt"
        outpath = os.path.join(root, file_out)
        if not os.path.exists(outpath):
            file_in = "train_sampled.txt"
            self.csv_data = pd.read_csv(os.path.join(root, file_in), sep='\t', index_col=None)

            cols = self.csv_data.columns.values
            for idx, col in enumerate(cols):
                le = LabelEncoder()
                le.fit(self.csv_data[col])
                self.csv_data[col] = le.transform(self.csv_data[col])

            self.csv_data.to_csv(outpath, sep='\t', index=False)
            print("Dataset processing completed.")

        self.csv_data = pd.read_csv(outpath, sep='\t', index_col=None)
        if train:
            global feature_sizes
            feature_sizes.clear()
            cols = self.csv_data.columns.values
            for col in cols:
                feature_sizes.append(len(self.csv_data[col].value_counts()))
            feature_sizes.pop(0)  # do not contain label

        self.train_data, self.test_data = train_test_split(self.csv_data, test_size=1/7, random_state=42)
    
    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)
    
    def __getitem__(self, idx):
        if self.train:
            x = self.train_data.iloc[idx].values
        else:
            x = self.test_data.iloc[idx].values
        x = np.array(x, dtype=np.float32)
        return x[1:], int(x[0]), idx


class AIDSDataset(Dataset):
    def __init__(self, train=True, root="./dataset", test_size=0.25, random_seed=42):
        super(AIDSDataset, self).__init__()
        self.root = root
        self.train = train

        # 下载 Abalone 数据集
        abalone = fetch_ucirepo(id=890)

        # 提取特征和目标变量
        X = abalone.data.features.values  # 转为 NumPy 数组
        y = abalone.data.targets.values.flatten()  # 确保 y 是 1D

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        # 根据 train 参数选择数据
        if self.train:
            self.X, self.y = self.X_train, self.y_train
        else:
            self.X, self.y = self.X_test, self.y_test

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.X)


class CDCDataset(Dataset):
    def __init__(self, train=True, root="./dataset", test_size=0.25, random_seed=42):
        super(CDCDataset, self).__init__()
        self.root = root
        self.train = train

        # 下载 CDC Diabetes Health Indicators 数据集
        cdc_diabetes_health_indicators = fetch_ucirepo(id=891)

        # 提取特征和目标变量
        X = cdc_diabetes_health_indicators.data.features.values  # 转为 NumPy 数组
        y = cdc_diabetes_health_indicators.data.targets.values.flatten()  # 确保 y 是 1D

        # 划分训练集和测试集
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_seed, stratify=y
        )

        # 根据 train 变量选择数据
        if self.train:
            self.X, self.y = self.X_train, self.y_train
        else:
            self.X, self.y = self.X_test, self.y_test

    def __getitem__(self, idx):
        features = self.X[idx]
        label = self.y[idx]

        return torch.tensor(features, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.X)



datasets_choices = [
    "mnist",
    "fashionmnist",
    "fmnist",
    "cifar10",
    "cifar100",
    "criteo",
    "cinic10",
    "aids",
    "cdc",
    "imagenet12"
]

datasets_name = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "fmnist":"FashionMNIST",
    "cifar10": "CIFAR10",
    "cifar100": "CIFAR100",
    "criteo": "Criteo",
    "cinic10": "CINIC10",
    "aids":"AIDS",
    "cdc":"CDC",
    "imagenet12":"ImageNet12",
}

datasets_dict = {
    "mnist": datasets.MNIST,
    "fashionmnist": datasets.FashionMNIST,
    "fmnist": datasets.FashionMNIST,
    "cifar10": datasets.CIFAR10,
    "cifar100": datasets.CIFAR100,
    "criteo": Criteo,
    "cinic10": CINIC10,
    "aids":AIDSDataset,
    "cdc":CDCDataset,
    "imagenet12":ImageNet12
}

datasets_classes = {
    "mnist": 10,
    "fashionmnist": 10,
    "fmnist": 10,
    "cifar10": 10,
    "cifar100": 100,
    "criteo": 2,
    "cinic10": 10,
    "aids":2,
    "cdc":2,
    "imagenet12":12
}

transforms_default = {
    "mnist": transforms.Compose([transforms.ToTensor()]),
    "fashionmnist": transforms.Compose([transforms.ToTensor()]),
    "fmnist": transforms.Compose([transforms.ToTensor()]),
    "cifar10": transforms.Compose([transforms.ToTensor()]),
    "cifar100": transforms.Compose([transforms.ToTensor()]),
    "criteo": None,
    "cinic10": transforms.Compose([transforms.ToTensor()]),
    "aids":None,
    "cdc":None,
    "imagenet12": transforms.Compose([transforms.ToTensor()]),
}

MEAN_IMAGENET = (0.485, 0.456, 0.406)
STD_IMAGENET  = (0.229, 0.224, 0.225)  

transforms_train_augment = {
    "mnist": transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ]),
    "fashionmnist": transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ]),
    "fmnist": transforms.Compose([
        # transforms.RandomCrop(28, padding=4),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.2860], std=[0.3530])
    ]),
    "cifar10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]),
    "cifar100": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    "criteo": None,
    "aids": None,
    "cdc": None,
    "cinic10": transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
    ]),
    
    "imagenet12": transforms.Compose([
                    transforms.CenterCrop(64),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])
    
}


        # self.transform_train = 
        
        # self.transform_test = transforms.Compose([
        #             # transforms.ToPILImage(),
        #             transforms.Resize(256),
        #             transforms.CenterCrop(224),
        #             transforms.ToTensor(),
        #             transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])


transforms_test_augment = {
    "mnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.1307], std=[0.3081])
    ]),
    "fashionmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "fmnist": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])
    ]),
    "cifar10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    ]),
    "cifar100": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
    ]),
    "criteo": None,
    "aids": None,
    "cdc": None,
    "cinic10": transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.47889522, 0.47227842, 0.43047404], std=[0.24205776, 0.23828046, 0.25874835])
    ]),
    "imagenet12": transforms.Compose([
                    transforms.CenterCrop(64),
                    transforms.ToTensor(),
                    transforms.Normalize(MEAN_IMAGENET, STD_IMAGENET)])
}