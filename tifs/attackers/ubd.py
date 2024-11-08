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
import time
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
        # predictions = top_model(agged_embeddings)
        predictions = top_model(None, None, agged_inputs=agged_embeddings, agged=True)

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
            
            # temp_predictions = top_model(agged_embeddings)
            temp_predictions = top_model(None, None, agged_inputs=agged_embeddings, agged=True)

            _, temp_predictions = torch.max(temp_predictions, dim=1)
            temp_predictions = temp_predictions.detach().cpu().tolist()

            for i in range(len(temp_indices_list)):
                indices = temp_indices_list[i]
                shuffled_predictions = predictions[indices]
                
                for j in range(len(shuffled_predictions)):
                    if shuffled_predictions[j] == temp_predictions[j]:
                        caps[temp_indices_list[i][j]][i] = caps[temp_indices_list[i][j]][i] + 1
        
        caps = np.array(caps,dtype=float)/repeat_times
        caps = np.round(caps, 1)
        
        return caps



    def calculate_cac_new(self, top_model, embeddings):
        feature_length = embeddings[0].size(1)
        client_num = len(embeddings)
        batch_size = embeddings[0].size(0)
        cacs = [[0] * client_num for _ in range(batch_size)]

        agged_embeds = self._aggregate(embeddings)
        agged_embeds = agged_embeds.to(self.device)

        for i in range(batch_size):
            temp = agged_embeds[i].unsqueeze(0)
            temp = temp.requires_grad_(True)
            temp.retain_grad()
            
            output = top_model(None, None, agged_inputs=temp, agged=True)
            class_idx = torch.argmax(output, dim=1)
            target_class_score = output[:, class_idx]
            
            top_model.zero_grad()
            target_class_score.backward(gradient=torch.ones_like(target_class_score), retain_graph=True)
            grad = temp.grad.flatten()
            # grad = F.relu(grad)
            
            temp = temp.flatten()
            for k in range(client_num):
                cacs[i][k] = torch.sum(temp[k*feature_length:(k+1)*feature_length]*grad[k*feature_length:(k+1)*feature_length]).cpu().item()
                
            temp.grad = None
            
        cacs_min = np.min(cacs, axis=1, keepdims=True)
        cacs_max = np.max(cacs, axis=1, keepdims=True)
        normalized_cacs = (cacs - cacs_min) / (cacs_max - cacs_min + 1e-8)
        # normalized_cacs = cacs
        normalized_cacs = np.round(normalized_cacs, 1)
        
        return normalized_cacs
    
    
    
    def backdoor_check(self, caps, cacs, active_id=1):
        batch_size  = len(caps)
        client_size = len(caps[0])
        
        backdoor_indices = []
        clean_indices    = []       
        
        for i in range(batch_size):
            caps[i][active_id] = 0
        
        for i in range(batch_size):
            for j in range(client_size):
                if caps[i][j]==1 and cacs[i][j] == 1:
                        backdoor_indices.append(i)

        for i in range(batch_size):
            if i not in backdoor_indices:
                clean_indices.append(i)

        indicator = np.zeros(batch_size)
        indicator[backdoor_indices] = 1
        indicator = indicator.astype(int)
        return backdoor_indices, clean_indices, indicator


    def classification_accuracy(self, pred, gt):
        acc = accuracy_score(gt, pred)
        acc = round(acc, 4)
        return acc


    def visualize_single_caps(self, caps, file_name, attacker, backdoor_indices): 
        caps = np.array(caps)
        indicator = np.zeros_like(caps)
        
        for j in range(len(indicator)):
            if j in backdoor_indices:
                indicator[j][attacker] = 1
        
        caps = caps.flatten()
        indicator = indicator.flatten()

        sample_indices = np.arange(len(caps))
        
        clean_indices = sample_indices[indicator == 0]
        backdoor_indices = sample_indices[indicator == 1]
        
        plt.clf()
        plt.figure(figsize=(10, 6))
        
        #c='5e9e6e'
        #c='b55d60'
        
        plt.ylim(-0.05, 1.05)
        plt.gca().set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
        
        plt.scatter(clean_indices, caps[clean_indices], c='#5975a4', marker='o', label='Clean')
        plt.scatter(backdoor_indices, caps[backdoor_indices], c='#cc8963', marker='o', label='Backdoor')

        plt.xlabel('Index', fontsize=26)
        plt.ylabel('CAP', fontsize=26)
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        
        # plt.gca().set_ylim(-0.02, 1.2)
        # plt.ylim(0,1)
        # plt.ylim(-0.1, 1.1)

        plt.legend(loc='center right', fontsize=26)

        file_path_pdf = f'{file_name}_cap.pdf'
        plt.tight_layout()
        plt.savefig(file_path_pdf)
                
        # plt.savefig(file_path_svg, bbox_inches='tight')
        # drawing = svg2rlg(file_path_svg)
        # file_path_pdf = f'{file_name}_cap.pdf'
        # renderPDF.drawToFile(drawing, file_path_pdf)
        # plt.close()
        # os.remove(file_path_svg)


    def visualize_single_cacs(self, cacs, file_name, attacker, backdoor_indices):
        cacs = np.array(cacs)
        indicator = np.zeros_like(cacs)
        
        for j in range(len(indicator)):
            if j in backdoor_indices:
                indicator[j][attacker] = 1
        
        cacs = cacs.flatten()
        indicator = indicator.flatten()

        sample_indices = np.arange(len(cacs))
        clean_indices = sample_indices[indicator == 0]
        backdoor_indices = sample_indices[indicator == 1]

        plt.clf()
        plt.figure(figsize=(10, 6))

        #c='5e9e6e'
        #c='b55d60'

        plt.ylim(-0.05, 1.05)
        plt.gca().set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])

        plt.scatter(clean_indices, cacs[clean_indices], c='#5975a4', marker='o', label='Clean')
        plt.scatter(backdoor_indices, cacs[backdoor_indices], c='#cc8963', marker='o', label='Backdoor')


        plt.xlabel('Index', fontsize=26)
        plt.ylabel('CAC', fontsize=26)
        
        
        plt.xticks(fontsize=24)
        plt.yticks(fontsize=24)
        
        plt.legend(loc='center right', fontsize=26)

        file_path_pdf = f'{file_name}_cac.pdf'
        plt.tight_layout()
        plt.savefig(file_path_pdf)
        
        
        # plt.savefig(file_path_svg, bbox_inches='tight')
        # drawing = svg2rlg(file_path_svg)
        # file_path_pdf = f'{file_name}_cac.pdf'
        # renderPDF.drawToFile(drawing, file_path_pdf)
        # plt.close()
        # os.remove(file_path_svg)



    def detect(self, image_path, idx, active_model, embeddings, labels, repeat_times, ground_truth=[], poison_indices=[]):
        start_time = time.time()
        cacs = self.calculate_cac_new(active_model, embeddings)
        caps = self.calculate_cap(active_model, embeddings, repeat_times=repeat_times)
        end_time = time.time()
        # 只画前5个Batch
        if idx < 5:
            self.visualize_single_caps(caps, f'{image_path}/{idx}', 0, poison_indices)
            self.visualize_single_cacs(cacs, f'{image_path}/{idx}', 0, poison_indices)
            pass
        b_idx, c_idx, indicator = self.backdoor_check(caps, cacs, active_id=1)
        acc = self.classification_accuracy(indicator, ground_truth)
        
        elapsed_time = end_time - start_time
        # print(f"T: {repeat_times}  运行时间: {elapsed_time:.6f} 秒")
        return acc, elapsed_time