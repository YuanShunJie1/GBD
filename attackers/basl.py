import torch
import numpy as np
import os
from .vflbase import BaseVFL
from ..ubd import UBDDefense
import torch.nn.functional as F

import random
import attackers.support as cc

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score, recall_score, f1_score

from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.metrics import confusion_matrix


def select(N=128, pr=0.5):
    sequence = np.arange(N)
    num_samples = int(pr * N)

    backdoor = np.random.choice(sequence, num_samples, replace=False)
    backdoor = backdoor.tolist()
    
    clean = []
    for i in range(N):
        if i not in backdoor:
            clean.append(i)
    
    indicator = np.zeros(N)
    indicator[backdoor] = 1
    return backdoor, clean, indicator



class TIFS(BaseVFL):
    def __init__(self, args, model, train_loader, test_loader, device, trigger):
        super(TIFS, self).__init__(args, model, train_loader, test_loader, device)
        self.args = args
        self.rate = 0.8
        self.auxiliary_number = 500
        self.auxiliary_index  = self.obtain_auxiliary_index()
        self.trigger = trigger

    def obtain_auxiliary_index(self):
        _all = []
        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            for i in range(len(labels)):
                if labels[i].item() == self.args.target_label:
                    _all.append(indices[i].item())
        
        return random.sample(_all, self.auxiliary_number)
    
    @torch.no_grad()
    def design_vec(self,):
        outputs = []
        self.model.train()
    
        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            # data = [temp for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]

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
        
            for i in self.args.attack_id:
                tmp_emb = self.model.passive[i](data[i])
                for j in range(len(labels)):
                    if indices[j] in self.auxiliary_index:
                        outputs.append(tmp_emb[j].detach().clone())

        outputs = torch.stack(outputs)
        target_clean_vecs = outputs.detach().cpu().numpy()
        dim = cc.filter_dim(target_clean_vecs)
        center = cc.cal_target_center(target_clean_vecs[dim].copy(), kernel_bandwidth=1000) 
        target_vec = cc.search_vec(center,target_clean_vecs)
        target_vec = target_vec.flatten()
        target_vec = torch.tensor(target_vec, requires_grad=True)
        target_vec = target_vec.to(self.device)
        return target_vec
    
    
    def train(self,):
        for epoch in range(self.args.epochs):
            if epoch >= self.args.attack_epoch:
                if self.trigger is None or epoch % 10 == 0:
                    self.trigger = self.design_vec()

            self.train_one(epoch)
            self.test()
            if epoch >= self.args.attack_epoch: self.backdoor()
            self.scheduler_entire.step()

            if epoch % 10 == 0:
                save_path = os.path.join('/home/shunjie/codes/tifs/imagenet12', f"model_dataset={self.args.dataset}_epoch={epoch}.pth")
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    #'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler_entire.state_dict(),
                    'trigger': self.trigger,
                    'args': self.args,
                }, save_path)
                print(f"Model saved at epoch {epoch} to {save_path}")



    def test(self):
        print("\n============== Test ==============")
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        test_loss = 0
        correct = 0
        num_iter = (len(self.test_loader.dataset)//(self.args.batch_size))+1
        with torch.no_grad():
            for i, (inputs, labels, indices) in enumerate(self.test_loader):
                # data, labels, index = batch_data
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)
                
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
                
                logits  = self.model(data)

                losses = F.cross_entropy(logits, labels, reduction='none')
                test_loss += torch.sum(losses).item()
                
                pred = logits.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()
        
        test_loss = test_loss / len(self.test_loader.dataset)
        
        test_acc = 100. * correct / len(self.test_loader.dataset)
        
        print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, len(self.test_loader.dataset), test_acc))        
        return test_acc


    def train_one(self, epoch):
        self.iteration = len(self.train_loader.dataset)
        self.model.train()

        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
            if self.args.dataset == 'cdc':
                # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                data = [inputs[:, :10].to(self.device), inputs[:, 10:].to(self.device)]
            elif self.args.dataset == 'aids':
                    # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]
            labels = labels.to(self.device)
            
            emb = []
            for i in range(self.args.num_passive):
                tmp_emb = self.model.passive[i](data[i])
                emb.append(tmp_emb)

            if epoch >= self.args.attack_epoch:
                condition = []
                for attacker in self.args.attack_id:
                    for i in range(len(labels)):
                        if indices[i].item() in self.auxiliary_index:
                            condition.append(i)
                    emb[attacker][condition].data = self.trigger

            # forward propagation
            agg_emb = self.model._aggregate(emb)
            logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
            loss = self.loss(logit, labels)                  
            
            self.optimizer_entire.zero_grad()
            loss.backward()
            self.optimizer_entire.step()

            # if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self.iteration:
            print('Epoch:{}/{}, Step:{} \tLoss: {:.6f}'.format(epoch+1, self.args.epochs, batch_idx+1, loss.item()))
        
        return 


    def backdoor(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels, indices) in enumerate(self.test_loader):
                
                if self.args.dataset == 'cdc':
                    # CDC 数据集有 20 个特征，前 10 个和后 10 个拆分
                    data = [inputs[:, :10].to(self.device), inputs[:, 10:].to(self.device)]
                elif self.args.dataset == 'aids':
                     # AIDS 数据集有 23个特征，前 12 个和后 11 个拆分
                    data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
                else:
                    # 其他情况，按照 dim=2 拆分
                    data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]            
                    
                targets = torch.tensor([self.args.target_label for i in range(len(indices))])
                targets = targets.to(self.device)

                emb = []
                for i in range(self.args.num_passive):
                    if i in self.args.attack_id:
                        triggers = self.trigger.repeat(len(inputs), 1)
                        triggers = triggers.to(self.device)
                        emb.append(triggers)
                    else:
                        tmp_emb = self.model.passive[i](data[i])
                        emb.append(tmp_emb)
                
                agg_emb = torch.cat(emb, dim=1)

                logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
                pred = logit.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.view_as(pred)).sum().item()
        
        test_acc = 100. * correct / len(self.test_loader.dataset)
        print('ASR: {}/{} ({:.2f}%)\n'.format(correct, len(self.test_loader.dataset), test_acc))        
        
        return test_acc



    def backdoor_detection(self, udb, dataloader, poisoning_rate, repeat_times=6, dataset='mnist', threshold=5.0):
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        all_ground_truth = []            
        all_indicator    = []

        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            if self.args.dataset == 'cdc':
                data = [inputs[:, :10].to(self.device), inputs[:, 10:].to(self.device)]
            elif self.args.dataset == 'aids':
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]   

            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            
            embeddings = []

            batch_size = len(indices)
            poison_size = int(poisoning_rate*batch_size)
            poison_indices = random.sample(list(range(batch_size)), poison_size)
            ground_truth = [1 if i in poison_indices else 0 for i in range(batch_size)]
            all_ground_truth.extend(ground_truth)

            for i in range(self.args.num_passive):
                if i in self.args.attack_id:
                    tmp_emb = self.model.passive[i](data[i])
                    tmp_emb[poison_indices] = self.trigger
                    tmp_emb = tmp_emb.to(self.device)
                    embeddings.append(tmp_emb)
                else:
                    tmp_emb = self.model.passive[i](data[i])
                    embeddings.append(tmp_emb)
            
            agg_emb = self.model._aggregate(embeddings)
            logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
            
            path = f'/home/shunjie/codes/tifs/image_dataset={dataset}_pr={poisoning_rate}_rt={repeat_times}_threshold={threshold}'
            if not os.path.exists(path): os.makedirs(path)
            
            indicator = udb.detect(path, batch_idx, self.model.active, embeddings, logit, repeat_times, ground_truth, poison_indices, threshold=threshold)
            
            all_indicator.extend(indicator)
    
        recall = recall_score(all_ground_truth, all_indicator)
        precision = precision_score(all_ground_truth, all_indicator)
        f1 = f1_score(all_ground_truth, all_indicator)

        return recall,precision,f1
    


    def backdoor_detection_single_cap(self, udb, dataloader, poisoning_rate, repeat_times=6, dataset='mnist', threshold=5.0):
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        all_ground_truth = []            
        all_indicator    = []

        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            if self.args.dataset == 'cdc':
                data = [inputs[:, :10].to(self.device), inputs[:, 10:].to(self.device)]
            elif self.args.dataset == 'aids':
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]   

            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            
            embeddings = []

            batch_size = len(indices)
            poison_size = int(poisoning_rate*batch_size)
            poison_indices = random.sample(list(range(batch_size)), poison_size)
            ground_truth = [1 if i in poison_indices else 0 for i in range(batch_size)]
            all_ground_truth.extend(ground_truth)

            for i in range(self.args.num_passive):
                if i in self.args.attack_id:
                    tmp_emb = self.model.passive[i](data[i])
                    tmp_emb[poison_indices] = self.trigger
                    tmp_emb = tmp_emb.to(self.device)
                    embeddings.append(tmp_emb)
                else:
                    tmp_emb = self.model.passive[i](data[i])
                    embeddings.append(tmp_emb)
            
            agg_emb = self.model._aggregate(embeddings)
            logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
            
            path = f'/home/shunjie/codes/tifs/image_dataset={dataset}_pr={poisoning_rate}_rt={repeat_times}_threshold={threshold}'
            if not os.path.exists(path): os.makedirs(path)
            
            indicator = udb.detect_single_cap(path, batch_idx, self.model.active, embeddings, logit, repeat_times, ground_truth, poison_indices, threshold=threshold)
            
            all_indicator.extend(indicator)
    
        recall = recall_score(all_ground_truth, all_indicator)
        precision = precision_score(all_ground_truth, all_indicator)
        f1 = f1_score(all_ground_truth, all_indicator)

        return recall,precision,f1
    

    def backdoor_detection_single_cac(self, udb, dataloader, poisoning_rate, repeat_times=6, dataset='mnist', threshold=5.0):
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        all_ground_truth = []            
        all_indicator    = []

        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            if self.args.dataset == 'cdc':
                data = [inputs[:, :10].to(self.device), inputs[:, 10:].to(self.device)]
            elif self.args.dataset == 'aids':
                data = [inputs[:, :12].to(self.device), inputs[:, 12:].to(self.device)]
            else:
                data = [temp.to(self.device) for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]   

            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            
            embeddings = []

            batch_size = len(indices)
            poison_size = int(poisoning_rate*batch_size)
            poison_indices = random.sample(list(range(batch_size)), poison_size)
            ground_truth = [1 if i in poison_indices else 0 for i in range(batch_size)]
            all_ground_truth.extend(ground_truth)

            for i in range(self.args.num_passive):
                if i in self.args.attack_id:
                    tmp_emb = self.model.passive[i](data[i])
                    tmp_emb[poison_indices] = self.trigger
                    tmp_emb = tmp_emb.to(self.device)
                    embeddings.append(tmp_emb)
                else:
                    tmp_emb = self.model.passive[i](data[i])
                    embeddings.append(tmp_emb)
            
            agg_emb = self.model._aggregate(embeddings)
            logit = self.model.active(None, None, agged_inputs=agg_emb, agged=True)
            
            path = f'/home/shunjie/codes/tifs/image_dataset={dataset}_pr={poisoning_rate}_rt={repeat_times}_threshold={threshold}'
            if not os.path.exists(path): os.makedirs(path)
            
            indicator = udb.detect_single_cac(path, batch_idx, self.model.active, embeddings, logit, repeat_times, ground_truth, poison_indices, threshold=threshold)
            
            all_indicator.extend(indicator)
    
        recall = recall_score(all_ground_truth, all_indicator)
        precision = precision_score(all_ground_truth, all_indicator)
        f1 = f1_score(all_ground_truth, all_indicator)

        return recall,precision,f1
    


    def backdoor_baseline_detection(self, dataloader, poisoning_rate, dataset='mnist', method='if'):
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        all_ground_truth = []            
        all_indicator    = []
        
        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            # data = [temp for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]     
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
            
            batch_size = len(indices)
            poison_size = int(poisoning_rate*batch_size)
            poison_indices = random.sample(list(range(batch_size)), poison_size)
            ground_truth = [1 if i in poison_indices else 0 for i in range(batch_size)]
            all_ground_truth.extend(ground_truth)

            for i in range(self.args.num_passive):
                if i in self.args.attack_id:
                    tmp_emb = self.model.passive[i](data[i])
                    tmp_emb[poison_indices] = self.trigger
            
            embeddings = np.array(tmp_emb.detach().cpu())
            
            if method == 'if':
                # if poisoning_rate == 0.8:
                #     contamination = 0.5
                # else:
                #     contamination = poisoning_rate
                scaler = StandardScaler()
                embeddings_scaled = scaler.fit_transform(embeddings)
                iso_forest = IsolationForest(random_state=42)
                iso_forest.fit(embeddings_scaled)
                predictions = iso_forest.predict(embeddings_scaled)
            if method == 'lof':
                # if poisoning_rate == 0.8:
                #     contamination = 0.5
                # else:
                #     contamination = poisoning_rate
                scaler = StandardScaler()
                embeddings_scaled = scaler.fit_transform(embeddings)
                lof = LocalOutlierFactor(n_neighbors=20,)
                predictions = lof.fit_predict(embeddings_scaled)
            if method == 'svm':
                scaler = StandardScaler()
                embeddings_scaled = scaler.fit_transform(embeddings)
                ocsvm = OneClassSVM(kernel="rbf", nu=0.5, gamma="scale")
                predictions = ocsvm.fit_predict(embeddings_scaled)
                    
            indicator = np.where(predictions == 1, 0, 1)
            all_indicator.extend(indicator)

        recall = recall_score(all_ground_truth, all_indicator)
        precision = precision_score(all_ground_truth, all_indicator)
        f1 = f1_score(all_ground_truth, all_indicator)

        return recall,precision,f1
    