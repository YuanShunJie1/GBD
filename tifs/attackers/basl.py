import torch
import numpy as np
import os
from .vflbase import BaseVFL
from .ubd import UBDDefense

import random
import helper as cc

from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest

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
            data = [temp for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]
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
            if epoch > self.args.attack_epoch: 
                self.backdoor()
            
            self.scheduler_entire.step()
            

    def train_one(self, epoch):
        self.iteration = len(self.train_loader.dataset)
        self.model.train()

        for batch_idx, (inputs, labels, indices) in enumerate(self.train_loader):
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

            if (batch_idx + 1) % 100 == 0 or (batch_idx + 1) == self.iteration:
                print('Epoch:{}/{}, Step:{} \tLoss: {:.6f}'.format(epoch+1, self.args.epochs, batch_idx+1, loss.item()))
        
        return 


    def backdoor(self):
        self.model.eval()
        correct = 0
        with torch.no_grad():
            for batch_idx, (inputs, labels, indices) in enumerate(self.test_loader):
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



    def backdoor_detection(self, udb, dataloader, poisoning_rate, repeat_times=6, dataset='mnist'):
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        correct = 0
        accs = []
        times = []
        
        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            data = [temp for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]            
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            
            embeddings = []

            batch_size = len(indices)
            poison_size = int(poisoning_rate*batch_size)
            poison_indices = random.sample(list(range(batch_size)), poison_size)
            ground_truth = [1 if i in poison_indices else 0 for i in range(batch_size)]

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
            
            if not os.path.exists(f'/home/shunjie/codes/tifs/image_{dataset}_{poisoning_rate}_{repeat_times}'):
                os.makedirs(f'/home/shunjie/codes/tifs/image_{dataset}_{poisoning_rate}_{repeat_times}')
            
            acc, elapsed_time = udb.detect(f'/home/shunjie/codes/tifs/image_{dataset}_{poisoning_rate}_{repeat_times}', batch_idx, self.model.active, embeddings, logit, repeat_times, ground_truth, poison_indices)
            
            accs.append(acc)
            times.append(elapsed_time)
        
        avd = round(sum(accs)/len(accs), 2)
        avd_time = round(sum(times)/len(accs), 2)

        return avd, avd_time
    
    


    def backdoor_baseline_detection(self, dataloader, poisoning_rate, dataset='mnist'):
        
        self.model.eval()
        self.model.active.eval()
        for i in range(self.args.num_passive):
            self.model.passive[i].eval()

        if_accs = []
        
        for batch_idx, (inputs, labels, indices) in enumerate(dataloader):
            data = [temp for temp in torch.chunk(inputs, self.args.num_passive, dim=2)]            
            for i in range(len(data)):
                data[i] = data[i].to(self.device)
            
            # embeddings = []

            batch_size = len(indices)
            poison_size = int(poisoning_rate*batch_size)
            poison_indices = random.sample(list(range(batch_size)), poison_size)
            ground_truth = [1 if i in poison_indices else 0 for i in range(batch_size)]

            for i in range(self.args.num_passive):
                if i in self.args.attack_id:
                    tmp_emb = self.model.passive[i](data[i])
                    tmp_emb[poison_indices] = self.trigger
            
            embeddings = np.array(tmp_emb.detach().cpu())
            
            scaler = StandardScaler()
            embeddings_scaled = scaler.fit_transform(embeddings)
            # embeddings_scaled = embeddings
            iso_forest = IsolationForest(contamination=0.5, random_state=42)
            iso_forest.fit(embeddings_scaled)
            predictions = iso_forest.predict(embeddings_scaled)
            indicator = np.where(predictions == 1, 0, 1)
            
            acc = accuracy_score(indicator, ground_truth)

            if_accs.append(acc)
        
        print(if_accs)
        avd = round(sum(if_accs)/len(if_accs), 2)
        
        return avd
    