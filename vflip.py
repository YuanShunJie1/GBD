import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch
from numbers import Number
from torch.autograd import Variable
import numpy as np
import random

torch.cuda.set_device(0)
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
device = torch.device(f'cuda:{0}')


class MAE(nn.Module):
    def __init__(self, input_dim):
        super(MAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2,input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(input_dim, input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2,input_dim*2),
            nn.ReLU(),
            nn.Linear(input_dim*2, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

class VFLIP:
    def __init__(self,args, model, device, dataset='mnist', p=2.0, num_passive=2, attack_id=[], auxiliary_index=[], trigger=None, d=128):
        self.args = args
        self.dataset = dataset
        self.p = p
        self.num_passive = num_passive
        self.attack_id = attack_id
        self.auxiliary_index = auxiliary_index
        self.trigger = trigger
        self.device = device
        self.model = model
        
        self.mae = MAE(input_dim=d).to(device)
        self.optimizer = torch.optim.Adam(self.mae.parameters(), lr=0.001)
        self.criterion = nn.MSELoss()
    
    def set_mae(self, mae, mean, std_dev):
        self.mae = mae
        self.mae = self.mae.to(device)
        self.mean = mean
        self.std_dev = std_dev
    
    def train_mae(self, train_loader):
        iteration = int(len(train_loader.dataset) / 128)
        self.mae.train()
        loss_box_epoch = []
        
        for epoch in range(100):
            for batch_idx, (inputs, labels, indices) in enumerate(train_loader):
                # data = [temp.to(self.device) for temp in torch.chunk(inputs, self.num_passive, dim=2)]
                
                if self.args.dataset == 'cdc':
                    # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                    data = [inputs[:, :10].to(self.device), inputs[:, 10:].to(self.device)]
                elif self.args.dataset == 'aids':
                        # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                    data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
                else:
                    # 其他情况，按照 dim=2 拆分
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]   

                for i in range(len(data)):
                    data[i] = data[i].to(self.device)
                    
                    labels = labels.to(self.device)
                    
                    emb = [self.model.passive[i](data[i]) for i in range(self.num_passive)]
                    
                    condition = [i for attacker in self.attack_id for i in range(len(labels)) if indices[i].item() in self.auxiliary_index]
                    for attacker in self.attack_id:
                        emb[attacker][condition].data = self.trigger
                    
                    h_clean = emb[1 - self.attack_id[0]]
                    h_backdoor = emb[self.attack_id[0]]
                    
                    reconstructed = self.mae(h_clean)
                    loss = self.criterion(reconstructed, h_backdoor)
                    
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()
                    
                    if epoch == 49:
                        loss_box_epoch.append(loss.item())
                    
                    print(f'Epoch:{epoch+1}/50, Step:{batch_idx+1}/{iteration} \tLoss: {loss.item():.6f}')
                    
        self.mean = np.mean(loss_box_epoch)
        self.std_dev = np.std(loss_box_epoch, ddof=0)
        return self.mean, self.std_dev


    def backdoor_detection(self, dataloader, poisoning_rate):
        self.model.train()
        self.model.active.train()
        
        for i in range(self.num_passive):
            self.model.passive[i].train()
        
        all_ground_truth = []            
        all_indicator    = []
        
        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            # data = [temp.to(self.device) for temp in torch.chunk(inputs, self.num_passive, dim=2)]
                
            if self.args.dataset == 'cdc':
                # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                data = [inputs[:, :10].to(self.device), inputs[:, 10:].to(self.device)]
            elif self.args.dataset == 'aids':
                    # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            else:
                # 其他情况，按照 dim=2 拆分
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]   

            
            batch_size = len(indices)
            poison_size = int(poisoning_rate * batch_size)
            poison_indices = random.sample(list(range(batch_size)), poison_size)
            ground_truth = [1 if i in poison_indices else 0 for i in range(batch_size)]
            
            all_ground_truth.extend(ground_truth)
            
            embeddings = []
            for i in range(self.num_passive):
                tmp_emb = self.model.passive[i](data[i])
                if i in self.attack_id:
                    tmp_emb[poison_indices] = self.trigger
                embeddings.append(tmp_emb.to(self.device))
            
            h_clean = embeddings[1 - self.attack_id[0]]
            h_backdoor = embeddings[self.attack_id[0]]
            
            reconstructed = self.mae(h_clean)
            indicator = []
            
            for j in range(inputs.size(0)):
                loss = F.mse_loss(reconstructed[j], h_backdoor[j], reduction='mean')
            
                if loss > self.mean + self.std_dev * self.p:
                    indicator.append(1)  # 预测为后门样本
                else:
                    indicator.append(0)  # 预测为良性样本

            all_indicator.extend(indicator)
        
        # 计算召回率、精确率和 F1 分数
        recall = recall_score(all_ground_truth, all_indicator)
        precision = precision_score(all_ground_truth, all_indicator)
        f1 = f1_score(all_ground_truth, all_indicator)

        print(recall, precision, f1)
        
        return recall, precision, f1

