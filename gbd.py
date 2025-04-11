import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F

# General Backdoor Defense
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



class GBDDefense(object):
    def __init__(self,device):
        self.device = device

    def _aggregate(self, x):
        return torch.cat(x, dim=1)

    @torch.no_grad()
    def calculate_cap(self, top_model, embeddings, repeat_times=3):
        agged_embeddings = self._aggregate(embeddings)
        agged_embeddings = agged_embeddings.to(self.device)
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


    def calculate_cac(self, top_model, embeddings):
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
            
            temp = temp.flatten()
            for k in range(client_num):
                cacs[i][k] = F.relu(torch.sum(temp[k*feature_length:(k+1)*feature_length]*grad[k*feature_length:(k+1)*feature_length])).cpu().item()
            temp.grad = None
            
        cacs_max = np.max(cacs, axis=1, keepdims=True)
        cacs_sec_max = np.partition(cacs, -2, axis=1)[:, -2][:, np.newaxis]
        
        cacs_sec_max = cacs_sec_max + 1e-8
        rcs = cacs_max / cacs_sec_max
        rcs = rcs.flatten()
        return rcs, cacs
    
    
    def backdoor_check(self, caps, cacs, threshold=2.0, active_id=1):
        batch_size  = len(caps)
        client_size = len(caps[0])
        
        backdoor_indices = []
        clean_indices    = []       
        
        for i in range(batch_size):
            caps[i][active_id] = 0
        
        for i in range(batch_size):
            if cacs[i] >= threshold:
                for j in range(client_size):
                    if caps[i][j]==1:
                        backdoor_indices.append(i)
                        break

        for i in range(batch_size):
            if i not in backdoor_indices:
                clean_indices.append(i)

        indicator = np.zeros(batch_size)
        indicator[backdoor_indices] = 1
        indicator = indicator.astype(int)
        return backdoor_indices, clean_indices, indicator

    
    def backdoor_check_single_cap(self, caps, threshold=2.0, active_id=1):
        batch_size  = len(caps)
        client_size = len(caps[0])
        
        backdoor_indices = []
        clean_indices    = []       
        
        for i in range(batch_size):
            caps[i][active_id] = 0
        
        for i in range(batch_size):
            # if cacs[i] >= threshold:
            for j in range(client_size):
                if caps[i][j]==1:
                    backdoor_indices.append(i)
                    break

        for i in range(batch_size):
            if i not in backdoor_indices:
                clean_indices.append(i)

        indicator = np.zeros(batch_size)
        indicator[backdoor_indices] = 1
        indicator = indicator.astype(int)
        return backdoor_indices, clean_indices, indicator

    
    def backdoor_check_single_cac(self, cacs, threshold=2.0, active_id=1):
        batch_size  = len(cacs)
        # client_size = len(caps[0])
        
        backdoor_indices = []
        clean_indices    = []       
        
        # for i in range(batch_size):
        #     caps[i][active_id] = 0
        
        for i in range(batch_size):
            if cacs[i] >= threshold:
                # for j in range(client_size):
                #     if caps[i][j]==1:
                backdoor_indices.append(i)
                # break

        for i in range(batch_size):
            if i not in backdoor_indices:
                clean_indices.append(i)

        indicator = np.zeros(batch_size)
        indicator[backdoor_indices] = 1
        indicator = indicator.astype(int)
        return backdoor_indices, clean_indices, indicator

    def detect(self, image_path, idx, active_model, embeddings, labels, repeat_times, ground_truth=[], poison_indices=[], threshold=2.0):
        cacs, cac_row = self.calculate_cac(active_model, embeddings)
        caps = self.calculate_cap(active_model, embeddings, repeat_times=repeat_times)
        _, _, indicator = self.backdoor_check(caps, cacs, threshold=threshold, active_id=1)
        return indicator

    def detect_single_cap(self, image_path, idx, active_model, embeddings, labels, repeat_times, ground_truth=[], poison_indices=[], threshold=2.0):
        caps = self.calculate_cap(active_model, embeddings, repeat_times=repeat_times)
        _, _, indicator = self.backdoor_check_single_cap(caps, threshold=threshold, active_id=1)
        return indicator
    
    def detect_single_cac(self, image_path, idx, active_model, embeddings, labels, repeat_times, ground_truth=[], poison_indices=[], threshold=2.0):
        cacs, cac_row = self.calculate_cac(active_model, embeddings)
        _, _, indicator = self.backdoor_check_single_cac(cacs, threshold=threshold, active_id=1)
        return indicator