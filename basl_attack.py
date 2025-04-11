import argparse
import os
import importlib
# import utils.models_ours as models, utils.datasets as datasets
import utils.models as models
import utils.datasets as datasets

from torch.utils.data import DataLoader
import torchvision

import torch
import random
# from torch.utils.data import Dataset
# from attackers.ubd import UBDDefense
from attackers.basl import TIFS
# from attackers.icdm import ICDM

# from utils.models import *


torch.autograd.set_detect_anomaly(True)

class TempDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None):
        self.dataset = full_dataset
        self.transform = transform
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def __len__(self):
        return self.dataLen

class BackdoorTestDataset(torch.utils.data.Dataset):
    def __init__(self, full_dataset=None, transform=None, target_label=0):
        
        self.dataset = []
        self.transform = transform
        
        for i in range(len(full_dataset)):
            if full_dataset[i][1] != target_label:
                self.dataset.append([full_dataset[i][0], full_dataset[i][1]])
        
        self.dataLen = len(self.dataset)

    def __getitem__(self, index):
        image = self.dataset[index][0]
        label = self.dataset[index][1]
        if self.transform:
            image = self.transform(image)
        return image, label, index

    def __len__(self):
        return self.dataLen

    # "mnist",
    # "fashionmnist",
    # "cifar10",
    # "cifar100",
    # "criteo",
    # "cinic10"

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset',
                        help='the datasets for evaluation;',
                        type=str,
                        choices=datasets.datasets_choices,
                        default='imagenet12')
    parser.add_argument('--epochs',
                        help='the number of epochs;',
                        type=int,
                        default=100)
    parser.add_argument('--attack_epoch',
                        help='set epoch for attacking, greater than or equal to 2;',
                        type=int,
                        default=80)
    parser.add_argument('--batch_size',
                        help='batch size;',
                        type=int,
                        default=32)
    parser.add_argument('--lr_passive',
                        help='learning rate for the passive parties;',
                        type=float,
                        default=0.1)
    parser.add_argument('--lr_active',
                        help='learning rate for the active party;',
                        type=float,
                        default=0.1)
    parser.add_argument('--lr_attack',
                        help='learning rate for the attacker;',
                        type=float,
                        default=0.1)
    parser.add_argument('--attack_id',
                        help="the ID list of the attacker, like ``--attack_id 0 1'' for [0,1];",
                        nargs='*',
                        type=int,
                        default=[0])
    parser.add_argument('--num_passive',
                        help='number of passive parties;',
                        type=int,
                        default=2)
    parser.add_argument('--division',
                        help='choose the data division mode;',
                        type=str,
                        choices=['vertical', 'random', 'imbalanced'],
                        default='vertical')
    parser.add_argument('--round',
                        help='round for log;',
                        type=int,
                        default=0)
    parser.add_argument('--target_label',
                        help='target label, which aim to change to;',
                        type=int,
                        default=0)
    parser.add_argument('--trigger',
                        help='set trigger type;',
                        type=str,
                        default='tifs')

    parser.add_argument('--gpuid', type=int,  default=0)

    parser.add_argument('--repeat_times', type=int, default=3)

    parser.add_argument('--alpha', type=float, default=0.05, help='uap learning rate decay')

    parser.add_argument('--eps', type=float, default=1.0, help='uap clamp bound')

    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 70])
    args = parser.parse_args()

    torch.cuda.set_device(0)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    device = torch.device(f'cuda:{0}')
    
    # determine whether the arguments are legal or not
    # check the arguments
    
    # change the arguments to dictionary and print
    print('Arguments:')
    args_vars = vars(args)
    format_args = '\t%' + str(max([len(i) for i in args_vars.keys()])) + 's : %s'
    for pair in sorted(args_vars.items()): print(format_args % pair)

    # create a log directory
    dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    log_dir = os.path.join(dir, "log", args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    data_dir = os.path.join(dir, "data", args.dataset)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    dataset_path = os.path.join(dir, 'dataset')
    if args.dataset == "cinic10":
        data_train = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/train')
        temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "cdc":
        data_train = datasets.datasets_dict[args.dataset](train=True)
        temp_dataset = TempDataset(full_dataset=data_train)
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "aids":
        data_train = datasets.datasets_dict[args.dataset](train=True)
        temp_dataset = TempDataset(full_dataset=data_train)
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "imagenet12":
        data_train = datasets.datasets_dict[args.dataset](train=True)
        temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, download=True)
        temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)


    if args.dataset == "cinic10":
        data_test = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/test')
        temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
        dataloader_test = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "cdc":
        data_test = datasets.datasets_dict[args.dataset](train=False)
        temp_dataset = TempDataset(full_dataset=data_test)
        dataloader_test = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "aids":
        data_test = datasets.datasets_dict[args.dataset](train=False)
        temp_dataset = TempDataset(full_dataset=data_test)
        dataloader_test = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "imagenet12":
        data_test = datasets.datasets_dict[args.dataset](train=False)
        temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
        dataloader_test = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, download=True)
        temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
        dataloader_test = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)

    entire_model = models.entire[args.dataset](num_passive=args.num_passive)
    entire_model = entire_model.to(device)

    attacker = TIFS(args,entire_model,dataloader_train,dataloader_test,device,trigger=None)

    model_name = f'entire_model_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}.pth'
    
    trigger_name = f'entire_model_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}_trigger.pth'
    
    attacker.train()
    attacker.test()
    attacker.backdoor()
    
    if not os.path.exists(model_name):
        torch.save(entire_model, model_name)
        torch.save(attacker.trigger, trigger_name)






if __name__ == '__main__':
    main()