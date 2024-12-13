import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision

from dataloaders.SymAug import Augmentation

import os
import cv2
import random
from utils import *
from utils.base_utils import map_to_range, reverse_mapping, integer_to_one_hot, kendall_tau, calculate_mae

class SymnetDataset(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
        self.crop = crop
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.crop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], _, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, map_to_range(label)

class SymnetDatasetKeypoint(Dataset):
    def __init__(self, 
                 tsv_file, 
                 istrain = True,
                 im_size = 224):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "/workspace/MTP/Phase_2/CosmeticEvalTrain/Data/"
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data\\', '')
        image_path = image_path.replace('dataset/image_dir/TMH/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        x_min = int(np.min(landmarks_target[:, 0]))
        x_max = int(np.max(landmarks_target[:, 0]))
        
        y_min = int(np.min(landmarks_target[:, 1]))
        y_max = int(np.max(landmarks_target[:, 1]))
        
        image = image[y_min:y_max, x_min:x_max, :]
        landmarks_target[:, 0] -= x_min
        landmarks_target[:, 1] -= y_min
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        rp = np.array([1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        lp = np.array([0,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
        left_points = landmarks_target[lp]
        right_points = landmarks_target[rp]
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            left_points[:, 0] += left.shape[1] - midline
            right = image[:, midline : , :]
            right_points[:, 0] -= midline
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
            right_points[:, 0] -= midline
        landmarks_target[:, 0] = self.im_size*landmarks_target[:, 0] / image.shape[1]
        landmarks_target[:, 1] = self.im_size*landmarks_target[:, 1] / image.shape[0]
        
        left_points[:, 0] = self.im_size*left_points[:, 0]/left.shape[1]
        left_points[:, 1] = self.im_size*left_points[:, 1]/left.shape[0]
        
        right_points[:, 0] = self.im_size*right_points[:, 0]/right.shape[1]
        right_points[:, 1] = self.im_size*right_points[:, 1]/right.shape[0]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], left_points, _ = aug.process(image_dictionary["left"], left_points, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], right_points, _ = aug.process(image_dictionary["right"], right_points, landmarks_5pts, scale, center_w, center_h)
            
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        fourth_channel_image = np.zeros((1,self.im_size,self.im_size), dtype=np.float32)
        
        landmarks_target = np.where(landmarks_target >= self.im_size-1, self.im_size-1, landmarks_target)
        left_points = np.where(left_points >= self.im_size-1, self.im_size-1, left_points)
        right_points = np.where(right_points >= self.im_size-1, self.im_size-1, right_points)
        
        landmarks_target = np.where(landmarks_target <= 0, 0, landmarks_target)
        left_points = np.where(left_points <= 0, 0, left_points)
        right_points = np.where(right_points <= 0, 0, right_points)
        
        fourth_channel_image[0,np.floor(landmarks_target[:,1]).astype(int),np.floor(landmarks_target[:,0]).astype(int)] = 1
        
        fourth_channel_left = np.zeros((1,self.im_size,self.im_size), dtype=np.float32)
        fourth_channel_left[0,np.floor(left_points[:,1]).astype(int),np.floor(left_points[:,0]).astype(int)] = 1
        
        fourth_channel_right = np.zeros((1,self.im_size,self.im_size), dtype=np.float32)
        fourth_channel_right[0,np.floor(right_points[:,1]).astype(int),np.floor(right_points[:,0]).astype(int)] = 1
        
        image_dictionary["image"] = np.concatenate([image_dictionary["image"], fourth_channel_image], axis=0)
        image_dictionary["left"] = np.concatenate([image_dictionary["left"], fourth_channel_left], axis=0)
        image_dictionary["right"] = np.concatenate([image_dictionary["right"], fourth_channel_right], axis=0)
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, map_to_range(label)

class SymNetClassification(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop = False):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
#         self.scale = 1/127.5   
#         self.means=(127.5, 127.5, 127.5)
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.iscrop = crop
        self.im_size = im_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
#         print('landmarks_target :', landmarks_target)
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.iscrop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
#             print(right.shape, midline)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = 224, aug_prob = 0.5)
            image_dictionary["image"], _, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            
#         if self.transform:
#             image = self.transform(image)
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
#         print(image_dictionary["image"].shape, self.means[0])
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, label-1

class SymDatasetBase(Dataset):
    def __init__(self, tsv_file, istrain = True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../STAR/TMH/dataset/image_dir/TMH"
#         self.scale = 1/127.5   
#         self.means=(127.5, 127.5, 127.5)
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('dataset/image_dir/TMH/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
#         print('landmarks_target :', landmarks_target)
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
#             print(right.shape, midline)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (224, 224))
        image_dictionary["left"] = cv2.resize(left, (224, 224))
        image_dictionary["right"] = cv2.resize(right, (224, 224))
        
        
        if self.transform:
            aug = Augmentation(image_size = 224, aug_prob = 0.5)
            image_dictionary["image"], _, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            
#         if self.transform:
#             image = self.transform(image)
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
#         print(image_dictionary["image"].shape, self.means[0])
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, map_to_range(label)

class EightNetDataset(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
#         self.scale = 1/127.5   
#         self.means=(127.5, 127.5, 127.5)
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        
        self.class_1 = self.data[self.data.iloc[:, 4] == 1]
        self.class_2 = self.data[self.data.iloc[:, 4] == 2]
        self.class_3 = self.data[self.data.iloc[:, 4] == 3]
        self.class_4 = self.data[self.data.iloc[:, 4] == 4]
        self.im_size = im_size
        self.crop = crop
    
    def __len__(self):
        return len(self.class_4)
    
    def __getitem__(self, idx):
        
        sample_paths = []
        sample_kpts = []
        
        sample_paths.extend([self.class_4.iloc[idx, 1]])
        
        landmarks_target = self.class_4.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
        sample_kpts.extend([landmarks_target])
        
        class_1_samples = self.class_1.sample(n=1)
        sample_paths.extend([class_1_samples.iloc[0 ,1]])
        landmarks_target = class_1_samples.iloc[0, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
        sample_kpts.extend([landmarks_target])
        
        class_2_samples = self.class_2.sample(n=3)
        class_3_samples = self.class_3.sample(n=3)
        
        sample_paths.extend(class_2_samples.iloc[:, 1].to_list())
        sample_paths.extend(class_3_samples.iloc[:, 1].to_list())
        
        for i in range(3):
            landmarks_target = class_2_samples.iloc[i, 2]
            landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
            sample_kpts.extend([landmarks_target])
            
        for i in range(3):
            landmarks_target = class_3_samples.iloc[i, 2]
            landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
            sample_kpts.extend([landmarks_target])
        
        aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
        
        idx = 0
        
        scale = 1.0
        center_w, center_h = self.im_size//2, self.im_size//2
        
        big_boy = {}
        
        key_list = ['class_0', 
                    'class_3', 
                    'class_1_1', 
                    'class_1_2', 
                    'class_1_3', 
                    'class_2_1', 
                    'class_2_2', 
                    'class_2_3']
        
        for image_path, landmarks_target in zip(sample_paths, sample_kpts):
            
            image_path = image_path.replace('Data/', '')
            image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
            image_path = os.path.join(self.image_dir, image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks_5pts = None
            if self.crop == True:
                x_min = int(np.min(landmarks_target[:, 0]))
                x_max = int(np.max(landmarks_target[:, 0]))

                y_min = int(np.min(landmarks_target[:, 1]))
                y_max = int(np.max(landmarks_target[:, 1]))

                image = image[y_min:y_max, x_min:x_max, :]
                landmarks_target[:, 0] -= x_min
                landmarks_target[:, 1] -= y_min
        
        
            midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
            midline = int(midline)
            left_size = (image.shape[0], midline, 3)
            right_size = (image.shape[0], image.shape[1] - midline, 3)

            if left_size < right_size:
                left = np.zeros(right_size)
                left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
                right = image[:, midline : , :]
            else:
                left = image[:, :midline, :]
                right = np.zeros(left_size)
                right[:, :image.shape[1] - midline, :] = image[:, midline:, :]

            image_dictionary = {}
            image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
            image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
            image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))

            center_w, center_h = self.im_size//2, self.im_size//2

            if self.transform:
                aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
                image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
                image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
                image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)

            image_dictionary["flattened_landmarks"] = landmarks_target.reshape((landmarks_target.shape[0]*landmarks_target.shape[1], ))/self.im_size
            image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
            image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
            image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0

            image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
            image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
            image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]

            image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
            image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
            image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]

            image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
            image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
            image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]

            image_dictionary["path"] = image_path
            image_dictionary["label"] = key_list[idx].split("_")[1]
            big_boy[key_list[idx]] = image_dictionary
            
            
            idx += 1
            
            
        
#             image_path = image_path.replace('Data/', '')
#             image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
#             image_path = os.path.join(self.image_dir, image_path)
            
#             image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             x_min = int(np.min(landmarks_target[:, 0]))
#             x_max = int(np.max(landmarks_target[:, 0]))

#             y_min = int(np.min(landmarks_target[:, 1]))
#             y_max = int(np.max(landmarks_target[:, 1]))

#             image = image[y_min:y_max, x_min:x_max, :]
#             landmarks_target[:, 0] -= x_min
#             landmarks_target[:, 1] -= y_min
            
#             image = cv2.resize(image, (self.im_size, self.im_size))
            
#             landmarks_target[:, 0] = self.im_size*landmarks_target[:, 0] / image.shape[1]
#             landmarks_target[:, 1] = self.im_size*landmarks_target[:, 1] / image.shape[0]
            
#             if self.transform:
#                 augmented_image, landmarks_target , _ = aug.process(image, landmarks_target, None, scale, center_w, center_h)
#             else:
#                 augmented_image = image
                
#             augmented_image = augmented_image.transpose(2, 0, 1).astype(np.float32)/255.0

#             augmented_image[0, :, :] = (augmented_image[0, :, :] - self.means[0]) * self.scale[0]
#             augmented_image[1, :, :] = (augmented_image[1, :, :] - self.means[1]) * self.scale[1]
#             augmented_image[2, :, :] = (augmented_image[2, :, :] - self.means[2]) * self.scale[2]
            
#             bigboy[idx] = torch.tensor(augmented_image)
#             idx += 1
        
        return big_boy

class SymnetDatasetNewOne(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "/workspace/MTP/Phase_2/CosmeticEvalTrain/Data/"
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
        self.crop = crop
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
#         image_path = image_path.replace('Data/', '')
#         image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        
        image_path = image_path.replace('Data\\', '')
        image_path = image_path.replace('dataset/image_dir/TMH/', '')
        
        if self.transform == True:
                
            if self.data.iloc[idx, 0] >= 951:
                # No need to append any path, image path is final
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:

                # CHose among black, white or asian
                random_flag = random.randint(1, 3)

                if 1:
                    # Original image
                    image_path = os.path.join(self.image_dir, image_path)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                elif random_flag == 2:
                    # Black image

                    # Read Original Image in case need to resize
                    I1 = cv2.imread(os.path.join(self.image_dir, image_path), cv2.IMREAD_COLOR)
                    image_path = image_path.replace('.jpg', '_BLACK.jpg')
                    image_path = os.path.join(self.image_dir, image_path)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (I1.shape[1], I1.shape[0]))


                else:
                    # White Image

                    # Read Original Image in case need to resize
                    I1 = cv2.imread(os.path.join(self.image_dir, image_path), cv2.IMREAD_COLOR)
                    image_path = image_path.replace('.jpg', '_WHITE.jpg')
                    image_path = os.path.join(self.image_dir, image_path)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (I1.shape[1], I1.shape[0]))
        else:
            image_path = os.path.join(self.image_dir, image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.crop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
        
        image_dictionary["flattened_landmarks"] = landmarks_target.reshape((landmarks_target.shape[0]*landmarks_target.shape[1], ))/self.im_size
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["path"] = image_path
        return image_dictionary, map_to_range(label)

class SymnetDatasetAPI(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
        self.crop = crop
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.crop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
        
        image_dictionary["flattened_landmarks"] = landmarks_target.reshape((landmarks_target.shape[0]*landmarks_target.shape[1], ))/self.im_size
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["path"] = image_path
        return image_dictionary, map_to_range(label)



import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import torchvision
from dataloaders.SymAug import Augmentation
import os
import cv2
import random
from utils import *

class SymnetDataset(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
        self.crop = crop
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.crop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], _, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, map_to_range(label)

class SymnetDatasetKeypoint(Dataset):
    def __init__(self, 
                 tsv_file, 
                 istrain = True,
                 im_size = 224,  image_directory="/workspace/MTP/Phase_2/CosmeticEvalTrain/Data/"):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = image_directory

        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data\\', '')
        image_path = image_path.replace('dataset/image_dir/TMH/', '')
#         print(f"Image path being read: {image_path}")
        image_path = os.path.join(self.image_dir, image_path)
#         print(f"Image path being read: {image_path}")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        x_min = int(np.min(landmarks_target[:, 0]))
        x_max = int(np.max(landmarks_target[:, 0]))
        
        y_min = int(np.min(landmarks_target[:, 1]))
        y_max = int(np.max(landmarks_target[:, 1]))
        
        image = image[y_min:y_max, x_min:x_max, :]
        landmarks_target[:, 0] -= x_min
        landmarks_target[:, 1] -= y_min
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        rp = np.array([1,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20])
        lp = np.array([0,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37])
        left_points = landmarks_target[lp]
        right_points = landmarks_target[rp]
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            left_points[:, 0] += left.shape[1] - midline
            right = image[:, midline : , :]
            right_points[:, 0] -= midline
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
            right_points[:, 0] -= midline
        landmarks_target[:, 0] = self.im_size*landmarks_target[:, 0] / image.shape[1]
        landmarks_target[:, 1] = self.im_size*landmarks_target[:, 1] / image.shape[0]
        
        left_points[:, 0] = self.im_size*left_points[:, 0]/left.shape[1]
        left_points[:, 1] = self.im_size*left_points[:, 1]/left.shape[0]
        
        right_points[:, 0] = self.im_size*right_points[:, 0]/right.shape[1]
        right_points[:, 1] = self.im_size*right_points[:, 1]/right.shape[0]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], left_points, _ = aug.process(image_dictionary["left"], left_points, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], right_points, _ = aug.process(image_dictionary["right"], right_points, landmarks_5pts, scale, center_w, center_h)
            
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        fourth_channel_image = np.zeros((1,self.im_size,self.im_size), dtype=np.float32)
        
        landmarks_target = np.where(landmarks_target >= self.im_size-1, self.im_size-1, landmarks_target)
        left_points = np.where(left_points >= self.im_size-1, self.im_size-1, left_points)
        right_points = np.where(right_points >= self.im_size-1, self.im_size-1, right_points)
        
        landmarks_target = np.where(landmarks_target <= 0, 0, landmarks_target)
        left_points = np.where(left_points <= 0, 0, left_points)
        right_points = np.where(right_points <= 0, 0, right_points)
        
        fourth_channel_image[0,np.floor(landmarks_target[:,1]).astype(int),np.floor(landmarks_target[:,0]).astype(int)] = 1
        
        fourth_channel_left = np.zeros((1,self.im_size,self.im_size), dtype=np.float32)
        fourth_channel_left[0,np.floor(left_points[:,1]).astype(int),np.floor(left_points[:,0]).astype(int)] = 1
        
        fourth_channel_right = np.zeros((1,self.im_size,self.im_size), dtype=np.float32)
        fourth_channel_right[0,np.floor(right_points[:,1]).astype(int),np.floor(right_points[:,0]).astype(int)] = 1
        
        image_dictionary["image"] = np.concatenate([image_dictionary["image"], fourth_channel_image], axis=0)
        image_dictionary["left"] = np.concatenate([image_dictionary["left"], fourth_channel_left], axis=0)
        image_dictionary["right"] = np.concatenate([image_dictionary["right"], fourth_channel_right], axis=0)
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, map_to_range(label)

#     Breakdown of image_dictionary
# The image_dictionary contains the following keys and their corresponding values:

# image:

# A tensor of shape (4, im_size, im_size), where:
# The first three channels (0, 1, 2) correspond to the RGB image after being resized and normalized (mean and scale applied).
# The fourth channel (3) is a binary channel that indicates the presence of keypoints (landmarks) from landmarks_target, where pixels corresponding to landmarks are set to 1, and all others are set to 0.
# left:

# Similar to image, but contains the left side of the image, also with shape (4, im_size, im_size). It includes:
# Three channels for the RGB representation of the left image.
# A fourth binary channel for the left landmarks.
# right:

# Similar to image and left, but contains the right side of the image, with shape (4, im_size, im_size). It includes:
# Three channels for the RGB representation of the right image.
# A fourth binary channel for the right landmarks.


class SymNetClassification(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop = False):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
#         self.scale = 1/127.5   
#         self.means=(127.5, 127.5, 127.5)
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.iscrop = crop
        self.im_size = im_size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
#         print('landmarks_target :', landmarks_target)
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.iscrop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
#             print(right.shape, midline)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = 224, aug_prob = 0.5)
            image_dictionary["image"], _, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            
#         if self.transform:
#             image = self.transform(image)
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
#         print(image_dictionary["image"].shape, self.means[0])
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, label-1

class SymDatasetBase(Dataset):
    def __init__(self, tsv_file, istrain = True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../STAR/TMH/dataset/image_dir/TMH"
#         self.scale = 1/127.5   
#         self.means=(127.5, 127.5, 127.5)
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('dataset/image_dir/TMH/', '')
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
#         print('landmarks_target :', landmarks_target)
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
#             print(right.shape, midline)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (224, 224))
        image_dictionary["left"] = cv2.resize(left, (224, 224))
        image_dictionary["right"] = cv2.resize(right, (224, 224))
        
        
        if self.transform:
            aug = Augmentation(image_size = 224, aug_prob = 0.5)
            image_dictionary["image"], _, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            
#         if self.transform:
#             image = self.transform(image)
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
#         print(image_dictionary["image"].shape, self.means[0])
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
    
        return image_dictionary, map_to_range(label)

class EightNetDataset(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
#         self.scale = 1/127.5   
#         self.means=(127.5, 127.5, 127.5)
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        
        self.class_1 = self.data[self.data.iloc[:, 4] == 1]
        self.class_2 = self.data[self.data.iloc[:, 4] == 2]
        self.class_3 = self.data[self.data.iloc[:, 4] == 3]
        self.class_4 = self.data[self.data.iloc[:, 4] == 4]
        self.im_size = im_size
        self.crop = crop
#     1. Initialization (__init__):
# tsv_file: The file containing the dataset information in a tab-separated format.
# istrain: A boolean flag indicating whether the dataset is used for training (if True, it will apply data augmentation).
# im_size: The size to which the images will be resized.
# crop: A boolean flag indicating whether the image should be cropped based on landmark keypoints.
# self.data: Reads the TSV file into a pandas DataFrame.
# self.means and self.scale: These are normalization parameters for the images. The means are the average pixel values for RGB channels, and the scale represents standard deviation inverses (used for standardizing the images).
# self.class_1, self.class_2, self.class_3, self.class_4: These are subsets of the dataset filtered based on the class label in the 5th column (column index 4). For example, self.class_1 contains all samples where the 5th column value is 1.
# Other Parameters:
# self.image_dir: Specifies the directory where the images are stored.
# self.im_size: The size for resizing images.
    
    def __len__(self):
        return len(self.class_4)
    
#     2. __len__:
# This method returns the length of self.class_4. This means the dataset will iterate over the samples from class 4 (suggesting that class 4 is the primary class of interest).
    
    def __getitem__(self, idx):
        
        sample_paths = []
        sample_kpts = []
        
        sample_paths.extend([self.class_4.iloc[idx, 1]])
        
        landmarks_target = self.class_4.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
        sample_kpts.extend([landmarks_target])
        
        class_1_samples = self.class_1.sample(n=1)
        sample_paths.extend([class_1_samples.iloc[0 ,1]])
        landmarks_target = class_1_samples.iloc[0, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
        sample_kpts.extend([landmarks_target])
        
        class_2_samples = self.class_2.sample(n=3)
        class_3_samples = self.class_3.sample(n=3)
        
        sample_paths.extend(class_2_samples.iloc[:, 1].to_list())
        sample_paths.extend(class_3_samples.iloc[:, 1].to_list())
        
        for i in range(3):
            landmarks_target = class_2_samples.iloc[i, 2]
            landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
            sample_kpts.extend([landmarks_target])
            
        for i in range(3):
            landmarks_target = class_3_samples.iloc[i, 2]
            landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(38, 2)
            sample_kpts.extend([landmarks_target])
        
#         sample_paths
#        [fv
#     'path/to/image1.jpg',  # from class_4
#     'path/to/image3.jpg',  # from class_1
#     'path/to/image4.jpg', 'path/to/image5.jpg', 'path/to/image6.jpg',  # from class_2
#     'path/to/image7.jpg', 'path/to/image8.jpg', 'path/to/image9.jpg'   # from class_3
#     ]



# sample_kpts[
#     array([[1.1, 2.2], [3.3, 4.4]], dtype=float32),  # from class_4
#     array([[9.9, 10.10], [11.11, 12.12]], dtype=float32),  # from class_1
#     array([[13.13, 14.14], [15.15, 16.16]], dtype=float32),  # from class_2
#     array([[17.17, 18.18], [19.19, 20.20]], dtype=float32),
#     array([[21.21, 22.22], [23.23, 24.24]], dtype=float32),
#     array([[25.25, 26.26], [27.27, 28.28]], dtype=float32),  # from class_3
#     array([[29.29, 30.30], [31.31, 32.32]], dtype=float32),
#     array([[33.33, 34.34], [35.35, 36.36]], dtype=float32)
# ]
        
#         Sample Paths and Keypoints:
# The method begins by collecting the image paths and their corresponding landmarks (keypoints). It does so for:
# One sample from class_4 (the current index),
# One randomly selected sample from class_1,
# Three randomly selected samples from both class_2 and class_3.
# The landmarks (keypoints) are parsed from the CSV file and reshaped into 38 pairs of 2D coordinates (38 keypoints per image).

        
        aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
        
        idx = 0
        
        scale = 1.0
        center_w, center_h = self.im_size//2, self.im_size//2
        
        big_boy = {}
        
        key_list = ['class_0', 
                    'class_3', 
                    'class_1_1', 
                    'class_1_2', 
                    'class_1_3', 
                    'class_2_1', 
                    'class_2_2', 
                    'class_2_3']
        
        for image_path, landmarks_target in zip(sample_paths, sample_kpts):
            
            image_path = image_path.replace('Data/', '')
            image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
            image_path = os.path.join(self.image_dir, image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            landmarks_5pts = None
            if self.crop == True:
                x_min = int(np.min(landmarks_target[:, 0]))
                x_max = int(np.max(landmarks_target[:, 0]))

                y_min = int(np.min(landmarks_target[:, 1]))
                y_max = int(np.max(landmarks_target[:, 1]))

                image = image[y_min:y_max, x_min:x_max, :]
                landmarks_target[:, 0] -= x_min
                landmarks_target[:, 1] -= y_min
            
#             Cropping:
# If self.crop is True, the image is cropped based on the minimum and maximum coordinates of the landmarks.
        
            midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
            midline = int(midline)
            left_size = (image.shape[0], midline, 3)
            right_size = (image.shape[0], image.shape[1] - midline, 3)
            #now we got midline so with midline we broke image into 2 parts left half and right half 
            if left_size < right_size:
                left = np.zeros(right_size)
                left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
                right = image[:, midline : , :]
#                 If the left half is smaller, the following happens:
# A zero-filled array left is created with the size of the right half (right_size). This will act as a placeholder for padding.
# The original left half of the image is then copied into the zero-filled left array. The line:
    
            else:
                left = image[:, :midline, :]
                right = np.zeros(left_size)
                right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
#                 If the left half is larger or equal to the right half:
# The left half is directly taken from the original image without any modification (left = image[:, :midline, :]).
# A zero-filled array right is created with the size of the left half (left_size).
                
    
#                 Splitting Images:
# After cropping, the image is divided into two halves: a "left" half and a "right" half. This is done by computing the midline based on the x-coordinates of specific keypoints (20th and 37th keypoints).

            image_dictionary = {}
            image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
            image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
            image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))

            center_w, center_h = self.im_size//2, self.im_size//2

            if self.transform:
                aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
                image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
                image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
                image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            
#             If self.transform is True, data augmentation is applied.
# A custom Augmentation class is used, which likely includes techniques like flipping, rotation, scaling, or color changes. The process is applied to the original image (image), left half (left), and right half (right).
# The landmarks are also adjusted accordingly during the augmentation (e.g., when the image is rotated, the landmarks need to rotate too).

            image_dictionary["flattened_landmarks"] = landmarks_target.reshape((landmarks_target.shape[0]*landmarks_target.shape[1], ))/self.im_size
#     The landmarks_target (which are x, y coordinates of keypoints) are reshaped into a flat array and normalized by dividing by the image size (self.im_size). This normalization ensures the landmarks are in the range [0, 1], which helps with training stability.


            image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
            image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
            image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
            
#             The images (image, left, and right) are transformed from the usual shape of (height, width, channels) to (channels, height, width) using transpose(2, 0, 1). This is the format expected by most deep learning libraries, like PyTorch.
# The images are converted to floating-point numbers (np.float32) and normalized to the range [0, 1] by dividing by 255 (since pixel values range from 0 to 255).

            image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
            image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
            image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]

            image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
            image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
            image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]

            image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
            image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
            image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]

#             This section performs mean subtraction and scaling for each image channel (Red, Green, Blue). The images are normalized using precomputed means and scale values (these values could be from the dataset or similar to ImageNet dataset stats). This normalization improves training stability and ensures that the model receives inputs in a consistent format.
            
            image_dictionary["path"] = image_path
            image_dictionary["label"] = key_list[idx].split("_")[1]
            big_boy[key_list[idx]] = image_dictionary
            
            
            idx += 1
            
            
        
#             image_path = image_path.replace('Data/', '')
#             image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
#             image_path = os.path.join(self.image_dir, image_path)
            
#             image = cv2.imread(image_path, cv2.IMREAD_COLOR)
#             image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
#             x_min = int(np.min(landmarks_target[:, 0]))
#             x_max = int(np.max(landmarks_target[:, 0]))

#             y_min = int(np.min(landmarks_target[:, 1]))
#             y_max = int(np.max(landmarks_target[:, 1]))

#             image = image[y_min:y_max, x_min:x_max, :]
#             landmarks_target[:, 0] -= x_min
#             landmarks_target[:, 1] -= y_min
            
#             image = cv2.resize(image, (self.im_size, self.im_size))
            
#             landmarks_target[:, 0] = self.im_size*landmarks_target[:, 0] / image.shape[1]
#             landmarks_target[:, 1] = self.im_size*landmarks_target[:, 1] / image.shape[0]
            
#             if self.transform:
#                 augmented_image, landmarks_target , _ = aug.process(image, landmarks_target, None, scale, center_w, center_h)
#             else:
#                 augmented_image = image
                
#             augmented_image = augmented_image.transpose(2, 0, 1).astype(np.float32)/255.0

#             augmented_image[0, :, :] = (augmented_image[0, :, :] - self.means[0]) * self.scale[0]
#             augmented_image[1, :, :] = (augmented_image[1, :, :] - self.means[1]) * self.scale[1]
#             augmented_image[2, :, :] = (augmented_image[2, :, :] - self.means[2]) * self.scale[2]
            
#             bigboy[idx] = torch.tensor(augmented_image)
#             idx += 1
        
        return big_boy
#     Storing Results:
# The results for each image (original, left, and right sections), along with the flattened keypoints, are stored in a dictionary (image_dictionary).
# The images are converted from the typical (height, width, channels) format to the format expected by neural networks (channels, height, width).
# The dictionary includes the following:
# image: The resized and normalized original image.
# left: The left half of the image.
# right: The right half of the image.
# flattened_landmarks: The flattened keypoints.
# path: The file path to the image.
# label: The class label (derived from key_list).
# Return:
# All the information (images, landmarks, labels) for the current sample is returned as a dictionary (big_boy) containing entries for different classes (class_0, class_3, etc.).

# big_boy = {
#     'class_0': {'image': ..., 'left': ..., 'right': ..., 'flattened_landmarks': ..., 'path': ..., 'label': ...},
#     'class_1': {'image': ..., 'left': ..., 'right': ..., 'flattened_landmarks': ..., 'path': ..., 'label': ...},
#     ...
# }


class SymnetDatasetNewOneTest(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
        self.crop = crop
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        
        if 1+1 == 2:
                
            if self.data.iloc[idx, 0] >= 951:
                # No need to append any path, image path is final
                image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            else:

                # CHose among black, white or asian
                # random_flag = random.randint(1, 3)
                
                random_flag = 3 # White
                if random_flag == 1:
                    # Original image
                    image_path = os.path.join(self.image_dir, image_path)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)

                elif random_flag == 2:
                    # Black image

                    # Read Original Image in case need to resize
                    I1 = cv2.imread(os.path.join(self.image_dir, image_path), cv2.IMREAD_COLOR)
                    image_path = image_path.replace('.jpg', '_BLACK.jpg')
                    image_path = os.path.join(self.image_dir, image_path)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (I1.shape[1], I1.shape[0]))


                else:
                    # White Image

                    # Read Original Image in case need to resize
                    I1 = cv2.imread(os.path.join(self.image_dir, image_path), cv2.IMREAD_COLOR)
                    image_path = image_path.replace('.jpg', '_WHITE.jpg')
                    image_path = os.path.join(self.image_dir, image_path)
                    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
                    image = cv2.resize(image, (I1.shape[1], I1.shape[0]))
        else:
            image_path = os.path.join(self.image_dir, image_path)
            image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.crop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
        
        image_dictionary["flattened_landmarks"] = landmarks_target.reshape((landmarks_target.shape[0]*landmarks_target.shape[1], ))/self.im_size
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["path"] = image_path
        return image_dictionary, map_to_range(label)

class SymnetDatasetAPI(Dataset):
    def __init__(self, tsv_file, istrain = True, im_size = 224, crop=True):
        self.data = pd.read_csv(tsv_file, delimiter='\t', header=None)
        self.transform = istrain
        self.image_dir = "../Data/dataset/image_dir/TMH"
        self.means=(0.485, 0.456, 0.406)
        self.scale=(1/0.229, 1/0.224, 1/0.225)
        self.im_size = im_size
        self.crop = crop
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        image_path = self.data.iloc[idx, 1]
        image_path = image_path.replace('Data/', '')
        image_path = image_path.replace('../Data/dataset/image_dir/TMH/', '')
        
        image_path = os.path.join(self.image_dir, image_path)
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = int(self.data.iloc[idx, -1])
        landmarks_5pts = None
        landmarks_target = self.data.iloc[idx, 2]
        landmarks_target = np.array(list(map(float, landmarks_target.split(","))), dtype=np.float32).reshape(
            38, 2)
        scale = 1.0
        landmarks_5pts = None
        
        if self.crop == True:
            x_min = int(np.min(landmarks_target[:, 0]))
            x_max = int(np.max(landmarks_target[:, 0]))

            y_min = int(np.min(landmarks_target[:, 1]))
            y_max = int(np.max(landmarks_target[:, 1]))

            image = image[y_min:y_max, x_min:x_max, :]
            landmarks_target[:, 0] -= x_min
            landmarks_target[:, 1] -= y_min
        
        
        midline = (landmarks_target[20, 0] + landmarks_target[37, 0]) // 2
        midline = int(midline)
        left_size = (image.shape[0], midline, 3)
        right_size = (image.shape[0], image.shape[1] - midline, 3)
        
        if left_size < right_size:
            left = np.zeros(right_size)
            left[:, left.shape[1] - midline:, :] = image[:, :midline, :]
            right = image[:, midline : , :]
        else:
            left = image[:, :midline, :]
            right = np.zeros(left_size)
            right[:, :image.shape[1] - midline, :] = image[:, midline:, :]
        
        image_dictionary = {}
        image_dictionary["image"] = cv2.resize(image, (self.im_size, self.im_size))
        image_dictionary["left"] = cv2.resize(left, (self.im_size, self.im_size))
        image_dictionary["right"] = cv2.resize(right, (self.im_size, self.im_size))
        
        center_w, center_h = self.im_size//2, self.im_size//2
        
        if self.transform:
            aug = Augmentation(image_size = self.im_size, aug_prob = 0.5)
            image_dictionary["image"], landmarks_target, _ = aug.process(image_dictionary["image"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["left"], _, _ = aug.process(image_dictionary["left"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
            image_dictionary["right"], _, _ = aug.process(image_dictionary["right"], landmarks_target, landmarks_5pts, scale, center_w, center_h)
        
        image_dictionary["flattened_landmarks"] = landmarks_target.reshape((landmarks_target.shape[0]*landmarks_target.shape[1], ))/self.im_size
        image_dictionary["image"] = image_dictionary["image"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["left"] = image_dictionary["left"].transpose(2, 0, 1).astype(np.float32)/255.0
        image_dictionary["right"] = image_dictionary["right"].transpose(2, 0, 1).astype(np.float32)/255.0
        
        image_dictionary["image"][0, :, :] = (image_dictionary["image"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["image"][1, :, :] = (image_dictionary["image"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["image"][2, :, :] = (image_dictionary["image"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["left"][0, :, :] = (image_dictionary["left"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["left"][1, :, :] = (image_dictionary["left"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["left"][2, :, :] = (image_dictionary["left"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["right"][0, :, :] = (image_dictionary["right"][0, :, :] - self.means[0]) * self.scale[0]
        image_dictionary["right"][1, :, :] = (image_dictionary["right"][1, :, :] - self.means[1]) * self.scale[1]
        image_dictionary["right"][2, :, :] = (image_dictionary["right"][2, :, :] - self.means[2]) * self.scale[2]
        
        image_dictionary["path"] = image_path
        return image_dictionary, map_to_range(label)

