# UBD

import torch
# from utils.metrics import Metrics
import os
import numpy as np
# import utils.datasets as datasets
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from svglib.svglib import svg2rlg
from reportlab.graphics import renderPDF
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

from sklearn.metrics import accuracy_score

# Unversial Backdoor Defense
# Class Activation Probability  (CAP)
# Class Activation Contribution (CAC)

def shuffle_embeddings_by_feature(embeddings, feature_length):
    batch_size, total_dim = embeddings.shape
    
    num_features = total_dim // feature_length
    reshaped_embeddings = embeddings.view(batch_size, num_features, feature_length)
    
    for i in range(num_features):
        idx = torch.randperm(batch_size)
        reshaped_embeddings[:, i, :] = reshaped_embeddings[idx, i, :]
    
    shuffled_embeddings = reshaped_embeddings.view(batch_size, total_dim)
    return shuffled_embeddings

def shuffle_embeddings_list(embeddings_list):
    shuffled_list = []
    shuffle_indices_list = []
    
    for embeddings in embeddings_list:
        idx = torch.randperm(embeddings.size(0))
        shuffled_embeddings = embeddings[idx]
        shuffled_list.append(shuffled_embeddings)
        shuffle_indices_list.append(idx)
    
    return shuffled_list, shuffle_indices_list



class UBDDefense(object):
    def __init__(self,device):
        self.device = device

    def _aggregate(self, x):
        return torch.cat(x, dim=1)

    @torch.no_grad()
    def calculate_cap(self, top_model, embeddings, repeat_times=3):
        agged_embeddings = self._aggregate(embeddings)
        agged_embeddings = agged_embeddings.to(self.device)
        predictions = top_model(agged_embeddings)
        _, predictions = torch.max(predictions, dim=1)
        predictions = predictions.detach().cpu()

        feature_length = embeddings[0].size(1)
        client_num = len(embeddings)
        batch_size = embeddings[0].size(0)
        
        caps = [[0] * client_num for _ in range(batch_size)]
        
        for _ in range(repeat_times):
            temp_embeddings, temp_indices_list = shuffle_embeddings_list(embeddings)
            agged_embeddings = self._aggregate(temp_embeddings)
            agged_embeddings = agged_embeddings.to(self.device)
            
            temp_predictions = top_model(agged_embeddings)
            _, temp_predictions = torch.max(temp_predictions, dim=1)
            temp_predictions = temp_predictions.detach().cpu().tolist()

            for i in range(len(temp_indices_list)):
                indices = temp_indices_list[i]
                shuffled_predictions = predictions[indices]
                
                for j in range(len(shuffled_predictions)):
                    if shuffled_predictions[j] == temp_predictions[j]:
                        caps[temp_indices_list[i][j]][i] = caps[temp_indices_list[i][j]][i] + 1
        
        caps = np.array(caps,dtype=float)/repeat_times
        return caps


    def calculate_cac(self, top_model, embeddings, labels, criterion):
        feature_length = embeddings[0].size(1)
        client_num = len(embeddings)
        batch_size = embeddings[0].size(0)
        cacs = [[0] * client_num for _ in range(batch_size)]

        labels_tensor = labels.to(self.device)

        agged_embeds = self._aggregate(embeddings)
        agged_embeds = agged_embeds.to(self.device)

        for i in range(batch_size):
            temp = agged_embeds[i].unsqueeze(0)
            temp = temp.requires_grad_(True)
            temp.retain_grad()
            
            temp_pred = top_model(temp)
            
            loss = criterion(temp_pred, labels_tensor[i].unsqueeze(0))
            loss.backward(retain_graph=True)
            grad = temp.grad.flatten()
            grad = F.relu(grad)

            for k in range(client_num):
                cacs[i][k] = torch.mean(grad[k*feature_length:(k+1)*feature_length]).cpu().item()
            temp.grad = None
            
        cacs_min = np.min(cacs, axis=1, keepdims=True)
        cacs_max = np.max(cacs, axis=1, keepdims=True)
        normalized_cacs = (cacs - cacs_min) / cacs_max
        return normalized_cacs
    
    
    def backdoor_check(self, caps, cacs, active_id=1):
        batch_size  = len(caps)
        client_size = len(caps[0])
        
        backdoor_indices = []
        clean_indices    = []       
        
        # 将主动参与特征的CAP置为0
        for i in range(batch_size):
            caps[i][active_id] = 0
        
        # 计算CAP!=1的样本的CAC均值作为threshold；
        _sum = 0
        _count = 0
        for i in range(batch_size):
            for j in range(client_size):    
                if caps[i][j] != 1:
                    _sum = _sum + cacs[i][j]
                    _count = _count + 1

        _threshold = round(_sum / _count, 2)
    
        # 比较CAP=1的样本中，CAC均值大于threshold，作为backdoor；
        for i in range(batch_size):
            for j in range(client_size):
                if caps[i][j]==1:
                    if cacs[i][j] > _threshold:
                        backdoor_indices.append(i)

        for i in range(batch_size):
            if i not in backdoor_indices:
                clean_indices.append(i)

        indicator = np.zeros(batch_size)
        indicator[backdoor_indices] = 1

        return backdoor_indices, clean_indices, indicator



    def detect(self, active_model, embeddings, labels, repeat_times):
        loss_func = torch.nn.CrossEntropyLoss()
        
        cacs = self.calculate_cac(active_model, embeddings, labels, loss_func)
        caps = self.calculate_cap(active_model, embeddings, repeat_times=repeat_times)
        
        b_idx, c_idx, indicator = self.backdoor_check(caps, cacs, active_id=1)
        
        return b_idx, c_idx, indicator
        # file_name = f'./images_{self.args.dataset}/epoch={epoch}_batch_idx={batch_idx}_cac_cap_times={self.args.repeat_times}'
        # self.visualize(caps, cacs, file_name, self.args.attack_id, temp_backdoor_index)
                
    

    def visualize(self, caps, cacs, file_name, who, backdoor_indices):
        caps = np.array(caps)
        cacs = np.array(cacs)
        
        indicator = np.zeros_like(caps)

        w, h = caps.shape
        
        for j in range(len(indicator)):
            if j in backdoor_indices:
                indicator[j][who] = 1
          
        caps = caps.flatten()
        cacs = cacs.flatten()  
        indicator = indicator.flatten() 

        clean_indices = np.where(indicator == 0)[0]
        backdoor_indices_flat = np.where(indicator == 1)[0]

        plt.clf()

        plt.figure(figsize=(8, 6))
        plt.scatter(caps[clean_indices], cacs[clean_indices], alpha=0.7, c='blue', edgecolors='k', label='Clean')
        plt.scatter(caps[backdoor_indices_flat], cacs[backdoor_indices_flat], alpha=0.9, c='red', edgecolors='k', label='Backdoor')
        
        plt.xlabel('CAP')
        plt.ylabel('CAC')
        plt.grid(True)
        # plt.show()
        
        plt.savefig(f'{file_name}.png', bbox_inches='tight')
        # drawing = svg2rlg(f'{file_name}.svg')
        # renderPDF.drawToFile(drawing, f'{file_name}.pdf')
        plt.close()
        # os.remove(f'{file_name}.svg')    

    def classification_accuracy(self, pred, gt):
        acc = accuracy_score(gt, pred)
        acc = round(acc, 4)
        return acc

    def cluster(self, cap_results, cac_results):
        cap_results = np.array(cap_results)
        cac_results = np.array(cac_results)
        
        w, h = cap_results.shape
        
        cap_results = cap_results.flatten()
        cac_results = cac_results.flatten()

        pass