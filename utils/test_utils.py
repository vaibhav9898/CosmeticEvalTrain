import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
from torch.nn import functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random
import pandas as pd
import torchvision
from dataloaders.SymAug import Augmentation
import os
import cv2
import wandb
from utils import *
from tqdm import tqdm
from sklearn.metrics import confusion_matrix
import seaborn as sns
from statistics import mode, StatisticsError
from utils.base_utils import map_to_range, reverse_mapping, integer_to_one_hot, kendall_tau, calculate_mae


def test_model_classification(model, dataloaders, device, image_datasets, criterion):

    kendalls_x = []
    kendalls_y = []

    running_loss = 0.0
    running_corrects = 0

    for inputs, labels in tqdm(dataloaders['validation']):
        labels = labels.to(device)

        outputs = model(inputs)
        loss = criterion(outputs, labels)

        _, preds_l = torch.max(outputs["joint_logits"], 1)
        _, preds_r = torch.max(outputs["joint_logits"], 1)
        _, preds_j = torch.max(outputs["joint_logits"], 1)

        preds = torch.mode(torch.stack([preds_l, preds_r, preds_j]), dim=0).values
        kendalls_x.extend(preds.tolist())
        kendalls_y.extend(labels.tolist())
        running_loss += loss.item() * inputs["image"].size(0)
        running_corrects += torch.sum(preds == labels.data)

    con_mat = confusion_matrix(kendalls_y, kendalls_x)
    
    epoch_loss = running_loss / len(image_datasets['validation'])
    epoch_acc = running_corrects.double() / len(image_datasets['validation'])
    kendalls_x = torch.tensor(kendalls_x)
    kendalls_y = torch.tensor(kendalls_y)
    print('{} loss: {:.4f}, acc: {:.4f}'.format('validation',
                                                epoch_loss,
                                                epoch_acc))
    kt = kendall_tau(kendalls_x, kendalls_y)
    mae = calculate_mae(kendalls_x, kendalls_y)
    print("Kendall's Tau : ", kt, " MAE : ", mae)
    print("Confusion Matrix : ")
    print(con_mat)

def test_model_regression(model, dataloaders, device, image_datasets, criterion):
    
    kendalls_x = []
    kendalls_y = []
    
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in tqdm(dataloaders['validation']):

        labels = labels.to(device)
        labels = labels.view((labels.shape[0], 1))
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        preds = torch.mean(torch.stack([outputs["joint_logits"], outputs["joint_logits"], outputs["joint_logits"]], dim=0), dim = 0)
        preds_shape = preds.shape[0]
        preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
        labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
        
        kendalls_x.extend(preds.tolist())
        kendalls_y.extend(labels.tolist())
        running_loss += loss.item() * inputs["image"].size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        
    con_mat = confusion_matrix(kendalls_y, kendalls_x)
    
        
    epoch_loss = running_loss / len(image_datasets['validation'])
    epoch_acc = running_corrects.double() / len(image_datasets['validation'])
    kendalls_x = torch.tensor(kendalls_x)
    kendalls_y = torch.tensor(kendalls_y)
    print('{} loss: {:.4f}, acc: {:.4f}'.format('validation',
                                                epoch_loss,
                                                epoch_acc))
    kt = kendall_tau(kendalls_x, kendalls_y)
    mae = calculate_mae(kendalls_x, kendalls_y)
    print("Kendall's Tau : ", kt, " MAE : ", mae)
    print("Confusion Matrix : ")
    print(con_mat)

def test_model_eightnet(model, dataloaders, device, image_datasets):
    kendalls_x = []
    kendalls_y = []
    
    running_loss = 0.0
    running_corrects = 0
    
    classes = [[], [], [], []]
    
    for inputs, labels in tqdm(dataloaders['validation']):

        labels = labels.to(device)
        #labels = labels.view((labels.shape[0], 1))
        outputs = model(inputs["image"])
        
        #print(int(3*labels.item()), outputs.item())
        classes[int(3*labels.item())].append(outputs.item())
    
    plt.figure(dpi=200)
    sns.violinplot(data=classes, inner="quartile")
            # Add labels and title
    plt.xlabel("List")
    plt.ylabel("Value")
    plt.title("Distribution of Different Lists")
            
    plt.show()
    plt.savefig("Infer_distribution.png")
    plt.close()

    ## Environment 
        
#         preds = torch.mean(torch.stack([outputs["left_logits"], outputs["right_logits"], outputs["joint_logits"]], dim=0), dim = 0)
#         preds_shape = preds.shape[0]
#         preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
#         labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
        
#         kendalls_x.extend(preds.tolist())
#         kendalls_y.extend(labels.tolist())
#         running_loss += loss.item() * inputs["left"].size(0)
#         running_corrects += torch.sum(preds == labels.data)
        
        
#     con_mat = confusion_matrix(kendalls_y, kendalls_x)
    
        
#     epoch_loss = running_loss / len(image_datasets['validation'])
#     epoch_acc = running_corrects.double() / len(image_datasets['validation'])
#     kendalls_x = torch.tensor(kendalls_x)
#     kendalls_y = torch.tensor(kendalls_y)
#     print('{} loss: {:.4f}, acc: {:.4f}'.format('validation',
#                                                 epoch_loss,
#                                                 epoch_acc))
#     kt = kendall_tau(kendalls_x, kendalls_y)
#     mae = calculate_mae(kendalls_x, kendalls_y)
#     print("Kendall's Tau : ", kt, " MAE : ", mae)
#     print("Confusion Matrix : ")
#     print(con_mat)

def test_combined_model_regression(model_1, model_2, model_3, model_4, dataloaders_224, dataloaders_256, device, image_datasets, criterion):
    
    kendalls_x1 = []
    kendalls_y1 = []
    
    kendalls_x2 = []
    kendalls_y2 = []
    
    kendalls_x3 = []
    kendalls_y3 = []
    
    kendalls_x4 = []
    kendalls_y4 = []
    
    kendalls_x5 = []
    kendalls_y5 = []
    
    running_loss = 0.0
    running_corrects = 0
    imp = {}
    
    for inputs, labels in tqdm(dataloaders_224['validation']):

        labels = labels.to(device)
        labels = labels.view((labels.shape[0], 1))
        
        outputs = model_1(inputs)
        loss = criterion(outputs, labels)

        preds = torch.mean(torch.stack([outputs["joint_logits"], outputs["joint_logits"], outputs["joint_logits"]], dim=0), dim = 0)
        preds_shape = preds.shape[0]
        preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
        labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
        
        kendalls_x1.extend(preds.tolist())
        kendalls_y1.extend(labels.tolist())
        running_loss += loss.item() * inputs["image"].size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        for p_idx in range(inputs["image"].size(0)):
            imp[inputs["path"][p_idx]] = [labels.tolist()[p_idx], preds.tolist()[p_idx]]
        
    
    for inputs, labels in tqdm(dataloaders_256['validation']):

        labels = labels.to(device)
        labels = labels.view((labels.shape[0], 1))
        outputs = model_2(inputs)
        loss = criterion(outputs, labels)

        preds = torch.mean(torch.stack([outputs["joint_logits"], outputs["joint_logits"], outputs["joint_logits"]], dim=0), dim = 0)
        preds_shape = preds.shape[0]
        preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
        labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
        
        kendalls_x2.extend(preds.tolist())
        kendalls_y2.extend(labels.tolist())
        running_loss += loss.item() * inputs["image"].size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        for p_idx in range(inputs["image"].size(0)):
            imp[inputs["path"][p_idx]].extend([labels.tolist()[p_idx], preds.tolist()[p_idx]])
        
        
    for inputs, labels in tqdm(dataloaders_224['validation']):

        labels = labels.to(device)
        labels = labels.view((labels.shape[0], 1))
        outputs = model_3(inputs)
        loss = criterion(outputs, labels)

        preds = torch.mean(torch.stack([outputs["joint_logits"], outputs["joint_logits"], outputs["joint_logits"]], dim=0), dim = 0)
        preds_shape = preds.shape[0]
        preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
        labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
        
        kendalls_x3.extend(preds.tolist())
        kendalls_y3.extend(labels.tolist())
        running_loss += loss.item() * inputs["image"].size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        for p_idx in range(inputs["image"].size(0)):
            imp[inputs["path"][p_idx]].extend([labels.tolist()[p_idx], preds.tolist()[p_idx]])
            
    for inputs, labels in tqdm(dataloaders_256['validation']):

        labels = labels.to(device)
        labels = labels.view((labels.shape[0], 1))
        outputs = model_4(inputs)
        loss = criterion(outputs, labels)

        preds = torch.mean(torch.stack([outputs["joint_logits"], outputs["joint_logits"], outputs["joint_logits"]], dim=0), dim = 0)
        preds_shape = preds.shape[0]
        preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
        labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
        
        kendalls_x4.extend(preds.tolist())
        kendalls_y4.extend(labels.tolist())
        running_loss += loss.item() * inputs["image"].size(0)
        running_corrects += torch.sum(preds == labels.data)
        
        for p_idx in range(inputs["image"].size(0)):
            imp[inputs["path"][p_idx]].extend([labels.tolist()[p_idx], preds.tolist()[p_idx]])
        
    kendalls_x = []
    kendalls_y = []
    kendalls_y1 = []
    kendalls_y2 = []
    kendalls_y3 = []
    kendalls_y4 = []
    
    for k,v in imp.items():
        assert v[0] == v[2] == v[4] == v[6]
        kendalls_x.append(v[0])
        kendalls_y1.append(v[1])
        kendalls_y2.append(v[3])
        kendalls_y3.append(v[5])
        kendalls_y4.append(v[7])
    
    # Iterate through the indices of the lists
    for i in range(len(kendalls_y1)):
        values = [kendalls_y1[i], kendalls_y2[i], kendalls_y3[i], kendalls_y4[i]]
        try:
            # Attempt to find the mode of the values
            common_mode = mode(values)
        except StatisticsError:
            # In case of no mode found (all three values different), use the value from kendalls_y1
            common_mode = kendalls_y1[i]
        # Append the mode or the fallback value to the result list
        kendalls_y.append(common_mode)
    
    
    con_mat = confusion_matrix(kendalls_y, kendalls_x)
    
        
#     epoch_loss = running_loss / len(image_datasets['validation'])
#     epoch_acc = running_corrects.double() / len(image_datasets['validation'])
    kendalls_x = torch.tensor(kendalls_x)
    kendalls_y = torch.tensor(kendalls_y)
    
    #### ABHIJEET INFERENCE LIST
    
    kt = kendall_tau(kendalls_x, kendalls_y)
    mae = calculate_mae(kendalls_x, kendalls_y)
    print("Kendall's Tau : ", kt, " MAE : ", mae)
    print("Confusion Matrix : ")
    print(con_mat)
    print("Accuracy : ", sum([1 for i,j in zip(kendalls_x,kendalls_y) if i==j]) / len(kendalls_x))
