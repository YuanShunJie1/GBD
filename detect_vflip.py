import argparse
import os
import importlib
# import utils.models_ours as models, utils.datasets as datasets
import utils.models as models, utils.datasets as datasets

from torch.utils.data import DataLoader
import torchvision

import torch
import random
from torch.utils.data import Dataset

from attackers.ubd import UBDDefense
from attackers.basl import TIFS
# from attackers.icdm import ICDM

from vflip import VFLIP

from sklearn.metrics import accuracy_score


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
                        default='cdc')
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
                        default=128)
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
                        default=4)
    parser.add_argument('--source_label',
                        help='source label, which aim to change;',
                        type=int,
                        default=1)
    parser.add_argument('--trigger',
                        help='set trigger type;',
                        type=str,
                        # choices=['badvfl', 'villain', 'badvfl', 'tifs', 'icdm'],
                        default='tifs')

    parser.add_argument('--gpuid', type=int,  default=0)

    parser.add_argument('--repeat_times', type=int, default=3)

    parser.add_argument('--alpha', type=float, default=0.05, help='uap learning rate decay')

    parser.add_argument('--eps', type=float, default=1.0, help='uap clamp bound')

    parser.add_argument('--schedule', type=int, nargs='+', default=[50, 70])

    parser.add_argument('--pr', type=float, default=0.5, help='poisoning_rate')

    parser.add_argument('--repeat', type=int,  default=3)


    args = parser.parse_args()

    torch.cuda.set_device(args.gpuid)
    random.seed(123)
    torch.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    device = torch.device(f'cuda:{args.gpuid}')
    
    # load dataset
    dir = "/".join(os.path.abspath(__file__).split("/")[:-1])
    
    dataset_path = os.path.join(dir, 'dataset')
    if args.dataset == "cinic10":
        data_train = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/train')
        temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    elif args.dataset == "cdc" or args.dataset == "aids":
        data_train = datasets.datasets_dict[args.dataset](train=True)
        temp_dataset = TempDataset(full_dataset=data_train)
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    else:
        data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, download=True)
        temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
        dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)

    if args.dataset == "cinic10":
        data_test = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/test')
        temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
        dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)

        test_dataset = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/test')
        backdoor_test_dataset = BackdoorTestDataset(test_dataset, transform=datasets.transforms_test_augment[args.dataset], target_label=0)
        dataloader_backdoor_test = DataLoader(backdoor_test_dataset, batch_size=64, shuffle=True)

    elif args.dataset == "cdc" or args.dataset == "aids":
        data_test = datasets.datasets_dict[args.dataset](train=False)
        temp_dataset = TempDataset(full_dataset=data_test)
        dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)        

        backdoor_test_dataset = BackdoorTestDataset(data_test, transform=datasets.transforms_test_augment[args.dataset], target_label=0)
        dataloader_backdoor_test = DataLoader(backdoor_test_dataset, batch_size=64, shuffle=True)

    else:
        data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, download=True)
        temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
        dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)
        
        backdoor_test_dataset = BackdoorTestDataset(data_test, transform=datasets.transforms_test_augment[args.dataset], target_label=0)
        dataloader_backdoor_test = DataLoader(backdoor_test_dataset, batch_size=64, shuffle=True)

    model_name = f'entire_model_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}.pth'
    
    trigger_name = f'entire_model_dataset={args.dataset}_epochs={args.epochs}_attack_epoch={args.attack_epoch}_batch_size={args.batch_size}_trigger={args.trigger}_attack_id={args.attack_id}_num_passive={args.num_passive}_target_label={args.target_label}_trigger.pth'
    
    
    entire_model = torch.load(model_name)
    entire_model = entire_model.to(device)

    trigger = torch.load(trigger_name)
    trigger = trigger.to(device)    

    attacker = TIFS(args,entire_model,dataloader_train,dataloader_test,device,trigger=trigger)

    attacker.test()
    attacker.backdoor()

    if args.dataset == 'mnist' or args.dataset == 'fmnist' or args.dataset == 'fashionmnist' or args.dataset == 'imagenet12':
        d = 128
    elif args.dataset == 'cifar10' or args.dataset == 'cinic10':
        d = 10
    else:
        d = 10
  
    if args.dataset.startswith('imagenet'):
        p = 2.5
    else:
        p = 2.0
    
    
    vflip = VFLIP(args, entire_model, device, dataset=args.dataset, p=p, num_passive=2, attack_id=args.attack_id, auxiliary_index=attacker.auxiliary_index, trigger=trigger, d=d)

    mae_name = f'mae_dataset={args.dataset}_p={p}_d={d}.pth'
    if os.path.exists(mae_name):
        mae = torch.load(mae_name)
        vflip.set_mae(mae['mae'],mae['mean'],mae['std_dev'])
    else:
        mean, std_dev = vflip.train_mae(dataloader_train)
        torch.save({'mae':vflip.mae,'mean':mean,'std_dev':std_dev}, mae_name)
    
    avd = vflip.backdoor_detection(dataloader_backdoor_test, args.pr)
    print('VFLIP','Dataset: ', args.dataset, ' Poisoning rate: ', args.pr, ' Metrics: ', avd)






if __name__ == '__main__':
    main()
    
    
    
    # if args.dataset == "cinic10":
    #     data_train = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/train')
    #     temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
    #     dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)
    # else:
    #     data_train = datasets.datasets_dict[args.dataset](dataset_path, train=True, download=True)
    #     temp_dataset = TempDataset(full_dataset=data_train, transform=datasets.transforms_train_augment[args.dataset])
    #     dataloader_train = DataLoader(temp_dataset, batch_size=args.batch_size, shuffle=True)

    # if args.dataset == "cinic10":
    #     data_test = torchvision.datasets.ImageFolder(dataset_path + '/CINIC10/test')
    #     temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_train_augment[args.dataset])
    #     dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)
    # else:
    #     data_test = datasets.datasets_dict[args.dataset](dataset_path, train=False, download=True)
    #     temp_dataset = TempDataset(full_dataset=data_test, transform=datasets.transforms_test_augment[args.dataset])
    #     dataloader_test = DataLoader(temp_dataset, batch_size=64, shuffle=True)
