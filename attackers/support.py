import torch
import torch.nn.functional as F
import numpy as np
import random
import math


MIN_DISTANCE = 0.000001  # mini error


def filter_dim(vecs, target_num=50):
    coef = np.corrcoef(vecs)
    rows = np.sum(coef,axis=1)
    selected = np.argpartition(rows,-target_num)[-target_num:]
    print(np.mean(np.corrcoef(vecs[selected])))
    return selected


def euclidean_dist(pointA, pointB):
    total = (pointA - pointB) * (pointA - pointB).T
    return math.sqrt(total)


def gaussian_kernel(distance, bandwidth):
    m = np.shape(distance)[0]  
    right = np.mat(np.zeros((m, 1)))  
    for i in range(m):
        right[i, 0] = (-0.5 * distance[i] * distance[i].T) / (bandwidth * bandwidth)
        right[i, 0] = np.exp(right[i, 0])
    left = 1 / (bandwidth * math.sqrt(2 * math.pi))

    gaussian_val = left * right
    return gaussian_val


def shift_point(point, points, kernel_bandwidth):
    points = np.mat(points)
    m = np.shape(points)[0] 
    point_distances = np.mat(np.zeros((m, 1)))
    for i in range(m):
        point_distances[i, 0] = euclidean_dist(point, points[i])

    point_weights = gaussian_kernel(point_distances, kernel_bandwidth) 
    all_sum = 0.0
    for i in range(m):
        all_sum += point_weights[i, 0]

    point_shifted = point_weights.T * points / all_sum
    return point_shifted


def group_points(mean_shift_points):
    group_assignment = []
    m, n = np.shape(mean_shift_points)
    index = 0
    index_dict = {}
    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        if item_1 not in index_dict:
            index_dict[item_1] = index
            index += 1

    for i in range(m):
        item = []
        for j in range(n):
            item.append(str(("%5.2f" % mean_shift_points[i, j])))

        item_1 = "_".join(item)
        group_assignment.append(index_dict[item_1])

    return group_assignment


def train_mean_shift(points, kenel_bandwidth=2):
    mean_shift_points = np.mat(points)
    max_min_dist = 1
    iteration = 0  
    m = np.shape(mean_shift_points)[0]  
    need_shift = [True] * m  

    while max_min_dist > MIN_DISTANCE:
        max_min_dist = 0
        iteration += 1
        for i in range(0, m):
            
            if not need_shift[i]:
                continue
            p_new = mean_shift_points[i]
            p_new_start = p_new
            p_new = shift_point(p_new, points, kenel_bandwidth)  
            dist = euclidean_dist(p_new, p_new_start)  

            if dist > max_min_dist:
                max_min_dist = dist
            if dist < MIN_DISTANCE:  
                need_shift[i] = False
            mean_shift_points[i] = p_new
    group = group_points(mean_shift_points)  
    return np.mat(points), mean_shift_points, group

def cal_target_center(target_clean_vecs,kernel_bandwidth):
    data = target_clean_vecs
    points, shift_points, cluster = train_mean_shift(data, kernel_bandwidth)
    center = np.array(shift_points[0])
    return center


def search_vec(center,target_clean_vecs):
    center = torch.tensor(center)
    target_clean_vecs = torch.tensor(target_clean_vecs)
    max_length = torch.max(torch.norm(target_clean_vecs,dim=1,p=2))
    target_vec = center * max_length/torch.norm(center,p=2)*30.0
    target_vec = target_vec.detach().flatten().cpu().numpy()
    return target_vec
