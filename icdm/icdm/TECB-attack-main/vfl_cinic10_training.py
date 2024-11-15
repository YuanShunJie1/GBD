import logging
import os
import random
import sys

import numpy as np
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
from fedml_core.model.baseline.vfl_models import BottomModelForCinic10, TopModelForCinic10
from fedml_core.trainer.vfl_trainer import VFLTrainer
from fedml_core.utils.utils import AverageMeter, keep_predict_loss, over_write_args_from_file, image_format_2_rgb
from fedml_core.data_preprocessing.CINIC10.dataset import CINIC10L


import torch
import torch.nn as nn
import argparse
import time
import glob
import shutil
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR100
from torch.utils.data import Subset
import logging



def save_checkpoint(state, filename):
    torch.save(state, filename)


def main(device, args):
    # random seed for 10 runs
    seed = 123
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #load data
    # Data normalization and augmentation (optional)
    transform = transforms.Compose([
        transforms.Lambda(image_format_2_rgb),
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.47889522, 0.47227842, 0.43047404),
                                (0.24205776, 0.23828046, 0.25874835))
    ])

    # Load CIFAR-10 dataset
    trainset = CINIC10L(root=args.data_dir, split='train', transform=transform)
    testset = CINIC10L(root=args.data_dir, split='test', transform=transform)

    # classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # 选择你感兴趣的类别
    # target_class = args.target_class

    # 找出这个类别的标签
    # target_label = classes.index(target_class)
    
    target_label = 0

    # 找出所有属于这个类别的样本的索引
    target_indices = np.where(np.array(trainset.targets) == target_label)[0]

    non_target_indices = np.where(np.array(testset.targets) != target_label)[0]

    non_target_set = Subset(testset, non_target_indices)

    # 从目标索引中随机选择10个索引
    selected_indices = np.random.choice(target_indices, args.poison_num, replace=False)

    train_queue = torch.utils.data.DataLoader(
        dataset=trainset,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers
    )
    test_queue = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size=args.batch_size,
        num_workers=args.workers
    )
    non_target_queue = torch.utils.data.DataLoader(
        dataset=non_target_set,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    # build model
    model_list = []
    model_list.append(BottomModelForCinic10())
    model_list.append(BottomModelForCinic10())
    model_list.append(TopModelForCinic10())

    # optimizer and stepLR
    optimizer_list = [
        torch.optim.SGD(model.parameters(), args.lr, momentum=args.momentum, weight_decay=args.weight_decay) for
        model
        in model_list
    ]

    stone1 = args.stone1  # 50 int(args.epochs * 0.5)
    stone2 = args.stone2  # 85 int(args.epochs * 0.8)
    lr_scheduler_top_model = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[2],  milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_a = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[0], milestones=[stone1, stone2], gamma=args.step_gamma)
    lr_scheduler_b = torch.optim.lr_scheduler.MultiStepLR(optimizer_list[1],  milestones=[stone1, stone2], gamma=args.step_gamma)
    # change the lr_scheduler to the one you want to use
    lr_scheduler_list = [lr_scheduler_a, lr_scheduler_b, lr_scheduler_top_model]

    vfltrainer = VFLTrainer(model_list)

    criterion = nn.CrossEntropyLoss().to(device)
    bottom_criterion = keep_predict_loss

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume, map_location=device)
            args.start_epoch = checkpoint['epoch']
            for i in range(len(model_list)):
                model_list[i].load_state_dict(checkpoint['state_dict'][i])
                # optimizer_list[i].load_state_dict(checkpoint['optimizer'][i])
            print("=> loaded checkpoint '{}' (epoch {})"
                    .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))


    print("################################ Train Federated Models ############################")
    best_asr = 0.0

    _, (x_val, y_val, index) = next(enumerate(train_queue))

    # delta = torch.zeros_like(x_val[1][1]).float().to(device)
    delta = torch.zeros((1, 3, x_val.shape[-2], args.half), device=device)
    delta.requires_grad_(True)

    # Set a 9-pixel pattern to 1
    #delta[:, 0:3, 0:3] = 1
    
    _best_score = 0
    
    for epoch in range(args.start_epoch, args.epochs):
        logging.info('epoch %d args.lr %e ', epoch, args.lr)

        if args.backdoor_start:
            if (epoch + 1) < args.backdoor:
                train_loss, delta = vfltrainer.train_narcissus(train_queue, criterion, bottom_criterion, optimizer_list, device, args, delta, selected_indices)
            elif (epoch + 1) >= args.backdoor and (epoch + 1) < args.poison_epochs:
                train_loss = vfltrainer.train_mul(train_queue, criterion, bottom_criterion, optimizer_list, device, args)
            else:
                train_loss = vfltrainer.train_poisoning(train_queue, criterion, bottom_criterion, optimizer_list, device, args, delta, selected_indices)
        else:
            train_loss = vfltrainer.train_mul(train_queue, criterion, bottom_criterion, optimizer_list,device, args)

        lr_scheduler_list[0].step()
        lr_scheduler_list[1].step()
        lr_scheduler_list[2].step()

        test_loss, top1_acc, top5_acc = vfltrainer.test_mul(test_queue, criterion, device, args)
        _, test_asr_acc, test_asr_acc5 = vfltrainer.test_backdoor_mul(non_target_queue, criterion, device, args, delta, target_label)


        print('| Epoch [%3d]  Train-loss: %.2f  Test-loss: %.2f  ACC: %.2f  ASR: %.2f'%(epoch+1, train_loss, test_loss, top1_acc, test_asr_acc))


        total_value = test_asr_acc + top1_acc
        is_best = (total_value >= _best_score)
        
        if is_best: _best_score = total_value

        save_model_dir = args.save + f"/saved_models"
        
        if not os.path.exists(save_model_dir): os.makedirs(save_model_dir)
        
        if is_best:
            if os.path.exists(os.path.join(save_model_dir, "checkpoint.pth")):
                os.remove(os.path.join(save_model_dir, "checkpoint.pth"))
                os.remove(os.path.join(save_model_dir, "delta.pth"))
            
            save_checkpoint({
                'epoch': epoch + 1,
                'best_auc': best_asr,
                'state_dict': [model_list[i].state_dict() for i in range(len(model_list))],
                'optimizer': [optimizer_list[i].state_dict() for i in range(len(optimizer_list))],
            }, os.path.join(save_model_dir, "checkpoint.pth"))
            
            torch.save(delta, os.path.join(save_model_dir, "delta.pth"))

    print("##################################test############################################")
    
    checkpoint_path = args.save + f"/saved_models" + '/checkpoint.pth'
    delta_path = args.save + f"/saved_models" + '/delta.pth'
    
    print("=> loading checkpoint '{}'".format(checkpoint_path))
    
    checkpoint = torch.load(checkpoint_path)
    
    for i in range(len(model_list)):
        model_list[i].load_state_dict(checkpoint['state_dict'][i])
        
    print("=> loaded checkpoint '{}' (epoch {})".format(checkpoint_path, checkpoint['epoch']))

    vfltrainer.update_model(model_list)
    delta = torch.load(delta_path)
    
    test_loss, top1_acc, top5_acc = vfltrainer.test_mul(test_queue, criterion, device, args)
    _, asr_top1_acc, _ = vfltrainer.test_backdoor_mul(non_target_queue, criterion, device, args, delta,target_label)
    
    print("################################ Main Task ############################")
    print(f"Epoch: {epoch}, test_loss: {round(test_loss,2)}, test_top1_acc: {round(top1_acc,2)}")

    print("################################ Backdoor Task ############################")
    print(f"Epoch: {epoch}, test_loss: {round(test_loss,2)}, test_asr_top1_acc: {round(asr_top1_acc,2)}")

    print("################################ End Task ############################")
    print("######################################################################")




if __name__ == '__main__':
    print("################################ Prepare Data ############################")

    # os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser("vflmodelnet")
    parser.add_argument('--data_dir', default="./data/CINIC-10/", help='location of the data corpus')
    parser.add_argument('-d', '--dataset', default='CINIC10L', type=str,
                        help='name of dataset',
                        choices=['CIFAR10', 'CIFAR100', 'TinyImageNet', 'CINIC10L', 'Yahoo', 'Criteo', 'BCW'])
    parser.add_argument('--name', type=str, default='vfl_CINIC10L', help='experiment name')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--lr', type=float, default=0.1, help='init learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--report_freq', type=float, default=10, help='report frequency')
    parser.add_argument('--workers', type=int, default=0, help='num of workers')
    parser.add_argument('--epochs', type=int, default=100, help='num of training epochs')
    parser.add_argument('--layers', type=int, default=18, help='total number of layers')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--grad_clip', type=float, default=5., help='gradient clipping')
    parser.add_argument('--gamma', type=float, default=0.97, help='learning rate decay')
    parser.add_argument('--decay_period', type=int, default=1, help='epochs between two learning rate decays')
    parser.add_argument('--parallel', action='store_true', default=False, help='data parallelism')
    parser.add_argument('--u_dim', type=int, default=64, help='u layer dimensions')
    parser.add_argument('--k', type=int, default=2, help='num of client')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('--save', default='./model/CINIC10/test/backdoor50_LRA_poison4_amp15', type=str,
                        metavar='PATH',
                        help='path to save checkpoint (default: none)')
    parser.add_argument('--step_gamma', default=0.1, type=float, metavar='S',
                        help='gamma for step scheduler')
    parser.add_argument('--stone1', default=50, type=int, metavar='s1',
                        help='stone1 for step scheduler')
    parser.add_argument('--stone2', default=85, type=int, metavar='s2',
                        help='stone2 for step scheduler')
    parser.add_argument('--half', help='half number of features, generally seen as the adversary\'s feature num. '
                                       'You can change this para (lower that party_num) to evaluate the sensitivity '
                                       'of our attack -- pls make sure that the model to be resumed is '
                                       'correspondingly trained.',
                        type=int,
                        default=16)  # choices=[16, 14, 32, 1->party_num]. CIFAR10-16, Liver-14, TinyImageNet-32

    parser.add_argument('--backdoor', type=float, default=20, help='backdoor frequency')
    parser.add_argument('--target_class', type=str, default='cat', help='backdoor target class')
    parser.add_argument('--alpha', type=float, default=0.01, help='uap learning rate decay')
    parser.add_argument('--eps', type=float, default=16 / 255, help='uap clamp bound')

    parser.add_argument('--marvell', action='store_true', default=False, help='marvell defense')
    parser.add_argument('--max_norm', action='store_true', default=False, help='maxnorm defense')
    parser.add_argument('--iso', action='store_true', default=False, help='iso defense')
    parser.add_argument('--gc', action='store_true', default=False, help='gc defense')
    parser.add_argument('--lap_noise', action='store_true', default=False, help='lap_noise defense')
    parser.add_argument('--signSGD', action='store_true', default=False, help='sign_SGD defense')

    parser.add_argument('--iso_ratio', type=float, default=0.01, help='iso defense ratio')
    parser.add_argument('--gc_ratio', type=float, default=0.01, help='gc defense ratio')
    parser.add_argument('--lap_noise_ratio', type=float, default=0.01, help='lap_noise defense ratio')

    parser.add_argument('--poison_num', type=int, default=100, help='num of poison data')
    parser.add_argument('--corruption_amp', type=float, default=10, help='amplication of corruption')
    parser.add_argument('--backdoor_start', action='store_true', default=False, help='backdoor')
    # config file
    parser.add_argument('--c', type=str, default='configs/base/cinic10_bestattack.yml')

    args = parser.parse_args()
    over_write_args_from_file(args, args.c)

    if not os.path.exists(args.save):
        os.makedirs(args.save)

    # 创建一个logger
    logger = logging.getLogger('experiment_logger')
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件
    fh = logging.FileHandler(args.save +'/experiment.log')
    fh.setLevel(logging.INFO)

    # 定义handler的输出格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    # 给logger添加handler
    logger.addHandler(fh)

    logger.info(args)
    logger.info(device)


    main(device=device, args=args)
    # reference training result:
    # --- epoch: 99, batch: 1547, loss: 0.11550658332804839, acc: 0.9359105089400196, auc: 0.8736984159409958
    # --- (0.9270889578726378, 0.5111934752243287, 0.5054099033579607, None)

    # --- epoch: 99, batch: 200, loss: 0.09191526211798191, acc: 0.9636565918783608, auc: 0.9552342451916291
    # --- (0.9754657898538487, 0.7605652456769234, 0.8317858679682943, None)

