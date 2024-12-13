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
# import wandb
from utils import *
from tqdm import tqdm
import seaborn as sns
from utils.base_utils import map_to_range, reverse_mapping, integer_to_one_hot, kendall_tau, calculate_mae


means=(0.485, 0.456, 0.406)
scales=(1/0.229, 1/0.224, 1/0.225)

def train_model_classification(model, criterion, optimizer, dataloaders, ckpt_tau_path, ckpt_accuracy_path, num_epochs, device, image_datasets, scheduler):
    best_kendall_tau = 0.0
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
#         wandb.log({
#             "Epoch" : epoch+1
#         })
        
        for phase in ['train', 'validation']:
            kendalls_x = []
            kendalls_y = []
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            rand_i = random.randint(0, len(dataloaders[phase])-1)
            idx_r = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
                labels = labels.to(device)

                outputs = model(inputs)
#                 print(outputs["left_logits"].shape, labels.shape)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                _, preds_l = torch.max(outputs["joint_logits"], 1)
                _, preds_r = torch.max(outputs["joint_logits"], 1)
                _, preds_j = torch.max(outputs["joint_logits"], 1)
                
                preds = torch.mode(torch.stack([preds_l, preds_r, preds_j]), dim=0).values
                kendalls_x.extend(preds.tolist())
                kendalls_y.extend(labels.tolist())
                running_loss += loss.item() * inputs["left"].size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if idx_r == rand_i:
                    l_i = inputs["image"][0].cpu().numpy()
                    r_i = l_i.copy()
                    j_i = l_i.copy()
                    
                    l_i[0, :, :] = means[0] + l_i[0, :, :]/scales[0]
                    l_i[1, :, :] = means[1] + l_i[1, :, :]/scales[1]
                    l_i[2, :, :] = means[2] + l_i[2, :, :]/scales[2]
                    
                    l_i = np.transpose(l_i, (1, 2, 0))*255
                    l_i = l_i.astype(np.int)
                    
                    r_i[0, :, :] = means[0] + r_i[0, :, :]/scales[0]
                    r_i[1, :, :] = means[1] + r_i[1, :, :]/scales[1]
                    r_i[2, :, :] = means[2] + r_i[2, :, :]/scales[2]

                    r_i = np.transpose(r_i, (1, 2, 0))*255
                    r_i = r_i.astype(np.int)
                    
                    j_i[0, :, :] = means[0] + j_i[0, :, :]/scales[0]
                    j_i[1, :, :] = means[1] + j_i[1, :, :]/scales[1]
                    j_i[2, :, :] = means[2] + j_i[2, :, :]/scales[2]

                    j_i = np.transpose(j_i, (1, 2, 0))*255
                    j_i = j_i.astype(np.int)
                    
                    plt.figure(dpi=200)
                    plt.subplot(131)
                    plt.imshow(l_i)
                    plt.axis("off")
                    plt.subplot(132)
                    plt.imshow(r_i)
                    plt.axis("off")
                    plt.subplot(133)
                    plt.imshow(j_i)
                    plt.axis("off")
                    plt.show()
                    # plt.savefig(f'./data/train_images/pretrained_pred_label_{epoch}.png')
#                     if phase == 'train':
#                         wandb.log({"Input After Augmentations" : plt})
#                     else:
#                         wandb.log({"Validation Input" : plt})
                    plt.close()
                idx_r += 1
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            kendalls_x = torch.tensor(kendalls_x)
            kendalls_y = torch.tensor(kendalls_y)
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
            kt = kendall_tau(kendalls_x, kendalls_y)
            mae = calculate_mae(kendalls_x, kendalls_y)
            print("Kendall's Tau : ", kt, " MAE : ", mae)
            
            if phase == 'train':
#                 wandb.log({
#                     "train_loss" : epoch_loss,
#                     "train_accuracy" : epoch_acc,
#                     "kendalls_tau_train" : kt,
#                     "train MAE" : mae
#                 })
                scheduler.step()
            else:
#                 wandb.log({
#                     "val_loss" : epoch_loss,
#                     "val_accuracy" : epoch_acc,
#                     "val_tau_train" : kt,
#                     "val MAE" : mae
#                 })
                pass
                
                if kt > best_kendall_tau:
                    best_kendall_tau = kt
                    best_epoch = epoch

                        # Save the best model
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_kendall_tau': kt,
        #                 }, 'Checkpoints/resnet_best_model_tau_110124.pth')
                    }, ckpt_tau_path)  
                if epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_epoch = epoch

                    # Save the best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_accuracy': best_accuracy,
        #             }, 'Checkpoints/resnet_best_model_accuracy_110124.pth')
                    }, ckpt_accuracy_path)
    return model

def train_model_regressionkpt(model, criterion, optimizer, dataloaders, ckpt_tau_path, ckpt_accuracy_path, num_epochs, device, image_datasets, scheduler):
    best_kendall_tau = 0.0
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
#         wandb.log({
#             "Epoch" : epoch+1
#         })
        
        for phase in ['train', 'validation']:
            kendalls_x = []
            kendalls_y = []
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            rand_i = random.randint(0, len(dataloaders[phase])-1)
            idx_r = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
#                 inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view((labels.shape[0], 1))
                outputs = model(inputs)
#                 print(outputs["left_logits"].dtype, labels.dtype)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

#                 _, preds_l = torch.max(outputs["left_logits"], 1)
#                 _, preds_r = torch.max(outputs["right_logits"], 1)
#                 _, preds_j = torch.max(outputs["joint_logits"], 1)
#                 preds = torch.mean(torch.stack([outputs["left_logits"], outputs["right_logits"], outputs["joint_logits"]], dim=0), dim = 0)
                preds = torch.mean(torch.stack([outputs["joint_logits"], outputs["joint_logits"], outputs["joint_logits"]], dim=0), dim = 0)
#                 print("Preds before : ", preds)
                preds_shape = preds.shape[0]
                preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
                labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
                
                #preds = torch.mode(torch.stack([preds_l, preds_r, preds_j]), dim=0).values
                
                kendalls_x.extend(preds.tolist())
                kendalls_y.extend(labels.tolist())
                running_loss += loss.item() * inputs["left"].size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if idx_r == rand_i:
                    l_i = inputs["left"][0].cpu().numpy()
                    r_i = inputs["right"][0].cpu().numpy()
                    j_i = inputs["image"][0].cpu().numpy()
#                     print(l_i.shape, r_i.shape, j_i.shape)
                    l_i[0, :, :] = means[0] + l_i[0, :, :]/scales[0]
                    l_i[1, :, :] = means[1] + l_i[1, :, :]/scales[1]
                    l_i[2, :, :] = means[2] + l_i[2, :, :]/scales[2]
                    
                    l_i = np.transpose(l_i, (1, 2, 0))*255
                    l_i = l_i.astype(int)
                    
                    r_i[0, :, :] = means[0] + r_i[0, :, :]/scales[0]
                    r_i[1, :, :] = means[1] + r_i[1, :, :]/scales[1]
                    r_i[2, :, :] = means[2] + r_i[2, :, :]/scales[2]

                    r_i = np.transpose(r_i, (1, 2, 0))*255
                    r_i = r_i.astype(int)
                    
                    j_i[0, :, :] = means[0] + j_i[0, :, :]/scales[0]
                    j_i[1, :, :] = means[1] + j_i[1, :, :]/scales[1]
                    j_i[2, :, :] = means[2] + j_i[2, :, :]/scales[2]

                    j_i = np.transpose(j_i, (1, 2, 0))*255
                    j_i = j_i.astype(int)
                    
                    plt.figure(dpi=200)
                    plt.subplot(231)
                    plt.imshow(l_i[:, :, :-1])
                    plt.axis("off")
                    plt.subplot(232)
                    plt.imshow(l_i[:, :, 3])
                    plt.axis("off")
                    plt.subplot(233)
                    plt.imshow(r_i[:, :, :-1])
                    plt.axis("off")
                    plt.subplot(234)
                    plt.imshow(r_i[:, :, 3])
                    plt.axis("off")
                    plt.subplot(235)
                    plt.imshow(j_i[:, :, :-1])
                    plt.axis("off")
                    plt.subplot(236)
                    plt.imshow(j_i[:, :, 3])
                    plt.axis("off")
                    plt.show()
                    # plt.savefig(f'./data/train_images/pretrained_pred_label_{epoch}.png')
#                     if phase == 'train':
#                         wandb.log({"Input After Augmentations" : plt})
#                     else:
#                         wandb.log({"Validation Input" : plt})
                    plt.close()
                idx_r += 1
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            kendalls_x = torch.tensor(kendalls_x)
            kendalls_y = torch.tensor(kendalls_y)
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
            kt = kendall_tau(kendalls_x, kendalls_y)
            mae = calculate_mae(kendalls_x, kendalls_y)
            print("Kendall's Tau : ", kt, " MAE : ", mae)
            
            if phase == 'train':
#                 wandb.log({
#                     "train_loss" : epoch_loss,
#                     "train_accuracy" : epoch_acc,
#                     "kendalls_tau_train" : kt,
#                     "train MAE" : mae
#                 })
#                 wandb.log({"LR" : optimizer.param_groups[0]["lr"]})
                scheduler.step()
                
            else:
#                 wandb.log({
#                     "val_loss" : epoch_loss,
#                     "val_accuracy" : epoch_acc,
#                     "val_tau_train" : kt,
#                     "val MAE" : mae
#                 })
                pass
        
            
            if kt > best_kendall_tau:
                best_kendall_tau = kt
                best_epoch = epoch

                    # Save the best model
                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_kendall_tau': kt,
    #                 }, 'Checkpoints/kpt_cropped_resnet_regression_best_model_tau_150124.pth')
                }, ckpt_tau_path)

            if epoch_acc > best_accuracy:
                best_accuracy = epoch_acc
                best_epoch = epoch

                # Save the best model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_accuracy': best_accuracy,
    #             }, 'Checkpoints/kpt_cropped_resnet_regression_best_model_accuracy_150124.pth')
                    }, ckpt_accuracy_path)

    return model

def train_model_regression(model, criterion, optimizer, dataloaders, ckpt_tau_path, ckpt_accuracy_path, num_epochs, device, image_datasets, scheduler):
    best_kendall_tau = 0.0
    best_accuracy = 0.0
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
#         wandb.log({
#             "Epoch" : epoch+1
#         })
        
        for phase in ['train', 'validation']:
            kendalls_x = []
            kendalls_y = []
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0
            
            rand_i = random.randint(0, len(dataloaders[phase])-1)
            idx_r = 0
            
            for inputs, labels in tqdm(dataloaders[phase]):
#                 inputs = inputs.to(device)
                labels = labels.to(device)
                labels = labels.view((labels.shape[0], 1))
                outputs = model(inputs)
#                 print(outputs["left_logits"].dtype, labels.dtype)
                loss = criterion(outputs, labels)

                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

#                 _, preds_l = torch.max(outputs["left_logits"], 1)
#                 _, preds_r = torch.max(outputs["right_logits"], 1)
#                 _, preds_j = torch.max(outputs["joint_logits"], 1)
                
#                 preds = torch.mean(torch.stack([outputs["left_logits"], outputs["right_logits"], outputs["joint_logits"]], dim=0), dim = 0)
                preds = torch.mean(torch.stack([outputs["joint_logits"], outputs["joint_logits"], outputs["joint_logits"]], dim=0), dim = 0)
#                 print("Preds before : ", preds)
                preds_shape = preds.shape[0]
                preds = reverse_mapping(preds).to(torch.int).view((preds_shape))
                labels = reverse_mapping(labels).to(torch.int).view((preds_shape))
                
                #preds = torch.mode(torch.stack([preds_l, preds_r, preds_j]), dim=0).values
                
                kendalls_x.extend(preds.tolist())
                kendalls_y.extend(labels.tolist())
                running_loss += loss.item() * inputs["image"].size(0)
                running_corrects += torch.sum(preds == labels.data)
                
                if idx_r == rand_i:
                    l_i = inputs["left"][0].cpu().numpy()
                    r_i = inputs["right"][0].cpu().numpy()
                    j_i = inputs["image"][0].cpu().numpy()
                    
                    l_i[0, :, :] = means[0] + l_i[0, :, :]/scales[0]
                    l_i[1, :, :] = means[1] + l_i[1, :, :]/scales[1]
                    l_i[2, :, :] = means[2] + l_i[2, :, :]/scales[2]
                    
                    l_i = np.transpose(l_i, (1, 2, 0))*255
                    l_i = l_i.astype(int)
                    
                    r_i[0, :, :] = means[0] + r_i[0, :, :]/scales[0]
                    r_i[1, :, :] = means[1] + r_i[1, :, :]/scales[1]
                    r_i[2, :, :] = means[2] + r_i[2, :, :]/scales[2]

                    r_i = np.transpose(r_i, (1, 2, 0))*255
                    r_i = r_i.astype(int)
                    
                    j_i[0, :, :] = means[0] + j_i[0, :, :]/scales[0]
                    j_i[1, :, :] = means[1] + j_i[1, :, :]/scales[1]
                    j_i[2, :, :] = means[2] + j_i[2, :, :]/scales[2]

                    j_i = np.transpose(j_i, (1, 2, 0))*255
                    j_i = j_i.astype(int)
                    
                    plt.figure(dpi=200)
                    plt.subplot(131)
                    plt.imshow(l_i)
                    plt.axis("off")
                    plt.subplot(132)
                    plt.imshow(r_i)
                    plt.axis("off")
                    plt.subplot(133)
                    plt.imshow(j_i)
                    plt.axis("off")
                    plt.show()
                    # plt.savefig(f'./data/train_images/pretrained_pred_label_{epoch}.png')
#                     if phase == 'train':
#                         wandb.log({"Input After Augmentations" : plt})
#                     else:
#                         wandb.log({"Validation Input" : plt})
                    plt.close()
                idx_r += 1
            epoch_loss = running_loss / len(image_datasets[phase])
            epoch_acc = running_corrects.double() / len(image_datasets[phase])
            kendalls_x = torch.tensor(kendalls_x)
            kendalls_y = torch.tensor(kendalls_y)
            print('{} loss: {:.4f}, acc: {:.4f}'.format(phase,
                                                        epoch_loss,
                                                        epoch_acc))
            kt = kendall_tau(kendalls_x, kendalls_y)
            mae = calculate_mae(kendalls_x, kendalls_y)
            print("Kendall's Tau : ", kt, " MAE : ", mae)
            
            if phase == 'train':
#                 wandb.log({
#                     "train_loss" : epoch_loss,
#                     "train_accuracy" : epoch_acc,
#                     "kendalls_tau_train" : kt,
#                     "train MAE" : mae
#                 })
#                 wandb.log({"LR" : optimizer.param_groups[0]["lr"]})
                scheduler.step()
            else:
#                 wandb.log({
#                     "val_loss" : epoch_loss,
#                     "val_accuracy" : epoch_acc,
#                     "val_tau_train" : kt,
#                     "val MAE" : mae
#                 })
                pass

                if kt > best_kendall_tau:
                    best_kendall_tau = kt
                    best_epoch = epoch

                        # Save the best model
                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_kendall_tau': kt,
                        }, ckpt_tau_path)

                if epoch_acc > best_accuracy:
                    best_accuracy = epoch_acc
                    best_epoch = epoch

                    # Save the best model
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_accuracy': best_accuracy,
                   }, ckpt_accuracy_path)

    return model

def train_model_eightnet(model, criterion, optimizer, dataloaders, ckpt_tau_path, ckpt_accuracy_path, num_epochs, device, image_datasets, scheduler):
    best_kendall_tau = float("inf")
    
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
#         wandb.log({
#             "Epoch" : epoch+1
#         })
        
        for phase in ['train', 'validation']:
            
            distribution_4 = []
            distribution_1 = []
            distribution_2 = []
            distribution_3 = []
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            
            rand_i = random.randint(0, len(dataloaders[phase])-1)
            idx_r = 0
            
            for inputs in tqdm(dataloaders[phase]):
                inputs = inputs.view(8,3,224,224)
                outputs = model(inputs)  
                outputs = outputs.view(8)
                loss = criterion(outputs)
                
                distribution_4.append(outputs[0])
                distribution_1.append(outputs[1])
                distribution_2.extend(outputs[2:5])
                distribution_3.extend(outputs[5:])
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()
                
                if idx_r == rand_i:
                    
                    c1 = inputs[0].cpu().numpy()
                    c4 = inputs[1].cpu().numpy()
                    
                    c21 = inputs[2].cpu().numpy()
                    c22 = inputs[3].cpu().numpy()
                    c23 = inputs[4].cpu().numpy()
                    
                    c31 = inputs[5].cpu().numpy()
                    c32 = inputs[6].cpu().numpy()
                    c33 = inputs[7].cpu().numpy()
                    
                    c1[0, :, :] = means[0] + c1[0, :, :]/scales[0]
                    c1[1, :, :] = means[1] + c1[1, :, :]/scales[1]
                    c1[2, :, :] = means[2] + c1[2, :, :]/scales[2]

                    c1 = np.transpose(c1, (1, 2, 0))*255
                    c1 = c1.astype(int)
                    
                    c4[0, :, :] = means[0] + c4[0, :, :]/scales[0]
                    c4[1, :, :] = means[1] + c4[1, :, :]/scales[1]
                    c4[2, :, :] = means[2] + c4[2, :, :]/scales[2]

                    c4 = np.transpose(c4, (1, 2, 0))*255
                    c4 = c4.astype(int)
                    
                    c21[0, :, :] = means[0] + c21[0, :, :]/scales[0]
                    c21[1, :, :] = means[1] + c21[1, :, :]/scales[1]
                    c21[2, :, :] = means[2] + c21[2, :, :]/scales[2]

                    c21 = np.transpose(c21, (1, 2, 0))*255
                    c21 = c21.astype(int)
                    
                    c22[0, :, :] = means[0] + c22[0, :, :]/scales[0]
                    c22[1, :, :] = means[1] + c22[1, :, :]/scales[1]
                    c22[2, :, :] = means[2] + c22[2, :, :]/scales[2]

                    c22 = np.transpose(c22, (1, 2, 0))*255
                    c22 = c22.astype(int)
                    
                    c23[0, :, :] = means[0] + c23[0, :, :]/scales[0]
                    c23[1, :, :] = means[1] + c23[1, :, :]/scales[1]
                    c23[2, :, :] = means[2] + c23[2, :, :]/scales[2]

                    c23 = np.transpose(c23, (1, 2, 0))*255
                    c23 = c23.astype(int)
                    
                    c31[0, :, :] = means[0] + c31[0, :, :]/scales[0]
                    c31[1, :, :] = means[1] + c31[1, :, :]/scales[1]
                    c31[2, :, :] = means[2] + c31[2, :, :]/scales[2]

                    c31 = np.transpose(c31, (1, 2, 0))*255
                    c31 = c31.astype(int)

                    c32[0, :, :] = means[0] + c32[0, :, :]/scales[0]
                    c32[1, :, :] = means[1] + c32[1, :, :]/scales[1]
                    c32[2, :, :] = means[2] + c32[2, :, :]/scales[2]

                    c32 = np.transpose(c32, (1, 2, 0))*255
                    c32 = c32.astype(int)

                    c33[0, :, :] = means[0] + c33[0, :, :]/scales[0]
                    c33[1, :, :] = means[1] + c33[1, :, :]/scales[1]
                    c33[2, :, :] = means[2] + c33[2, :, :]/scales[2]

                    c33 = np.transpose(c33, (1, 2, 0))*255
                    c33 = c33.astype(int)
                    
                    
                    plt.figure(dpi=200)
                    plt.subplot(331)
                    plt.imshow(c1)
                    plt.axis("off")
                    plt.subplot(332)
                    plt.imshow(c4)
                    plt.axis("off")
                    
                    plt.subplot(334)
                    plt.imshow(c21)
                    plt.axis("off")
                    plt.subplot(335)
                    plt.imshow(c22)
                    plt.axis("off")
                    plt.subplot(336)
                    plt.imshow(c23)
                    plt.axis("off")
                    
                    plt.subplot(337)
                    plt.imshow(c31)
                    plt.axis("off")
                    plt.subplot(338)
                    plt.imshow(c32)
                    plt.axis("off")
                    plt.subplot(339)
                    plt.imshow(c33)
                    plt.axis("off")
                    
                    
                    plt.show()
#                     if phase == 'train':
#                         wandb.log({"Input After Augmentations" : plt})
#                     else:
#                         wandb.log({"Validation Input" : plt})
                    plt.close()
                idx_r += 1
            
            data = [torch.tensor(distribution_1).cpu(), torch.tensor(distribution_2).cpu(), torch.tensor(distribution_3).cpu(), torch.tensor(distribution_4).cpu()]

            # Create a violin plot using Seaborn
            plt.figure(dpi=200)
            sns.violinplot(data=data, inner="quartile")
            # Add labels and title
            plt.xlabel("List")
            plt.ylabel("Value")
            plt.title("Distribution of Different Lists")
            
            plt.show()
#             if phase == 'train':
#                 wandb.log({"Train Scores distribution" : plt})
#             else:
#                 wandb.log({"Validation Scores distribution" : plt})
            plt.close()
            
            epoch_loss = running_loss / len(image_datasets[phase])
            
            print('{} loss: {:.4f}'.format(phase,epoch_loss))
            kt = epoch_loss
            
            if phase == 'train':
#                 wandb.log({
#                     "train_loss" : epoch_loss
#                 })
#                 wandb.log({"LR" : optimizer.param_groups[0]["lr"]})
                scheduler.step()
            else:
#                 wandb.log({
#                     "val_loss" : epoch_loss
#                 })
                pass

                if kt < best_kendall_tau:
                    best_kendall_tau = kt
                    best_epoch = epoch

                    torch.save({
                            'epoch': epoch,
                            'model_state_dict': model.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'best_loss': kt,
                        }, ckpt_tau_path)
                    
    return model

def train_model_eightnet_latest(model, criterion, optimizer, dataloaders, ckpt_tau_path, ckpt_accuracy_path, num_epochs, device, image_datasets, scheduler):
    best_kendall_tau = float("inf")
    
    key_list = [
                    'class_0', 
                    'class_3', 
                    'class_1_1', 
                    'class_1_2', 
                    'class_1_3', 
                    'class_2_1', 
                    'class_2_2', 
                    'class_2_3'
                ]
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)
#         wandb.log({
#             "Epoch" : epoch+1
#         })
        
        for phase in ['train', 'validation']:
            
            distribution_4 = []
            distribution_1 = []
            distribution_2 = []
            distribution_3 = []
            
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            
            rand_i = random.randint(0, len(dataloaders[phase])-1)
            idx_r = 0
            
            for inputs in tqdm(dataloaders[phase]):
                
                input_T = {
                    'left' : torch.zeros((8,3,224,224)),
                    'right' : torch.zeros((8,3,224,224)),
                    'image' : torch.zeros((8,3,224,224)),
                    'flattened_landmarks' : torch.zeros((8, 76))
                }
                
                for tidx in range(8):
                    input_T['left'][tidx] = inputs[key_list[tidx]]['left']
                    input_T['right'][tidx] = inputs[key_list[tidx]]['right']
                    input_T['image'][tidx] = inputs[key_list[tidx]]['image']
                    input_T['flattened_landmarks'][tidx] = inputs[key_list[tidx]]['flattened_landmarks']
                
#                 inputs = inputs.view(8,3,224,224)
                outputs = model(input_T)['joint_logits'] 
                
                loss = criterion(outputs)
                
                distribution_4.append(outputs[0])
                distribution_1.append(outputs[1])
                distribution_2.extend(outputs[2:5])
                distribution_3.extend(outputs[5:])
                
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
                running_loss += loss.item()
                
                idx_r += 1
            
            data = [torch.tensor(distribution_1).cpu(), torch.tensor(distribution_2).cpu(), torch.tensor(distribution_3).cpu(), torch.tensor(distribution_4).cpu()]

            # Create a violin plot using Seaborn
            plt.figure(dpi=200)
            sns.violinplot(data=data, inner="quartile")
            # Add labels and title
            plt.xlabel("List")
            plt.ylabel("Value")
            plt.title("Distribution of Different Lists")
            
            plt.show()
#             if phase == 'train':
#                 wandb.log({"Train Scores distribution" : plt})
#             else:
#                 wandb.log({"Validation Scores distribution" : plt})
            plt.close()
            
            epoch_loss = running_loss / len(image_datasets[phase])
            
            print('{} loss: {:.4f}'.format(phase,epoch_loss))
            kt = epoch_loss
            
#             if phase == 'train':
#                 wandb.log({
#                     "train_loss" : epoch_loss
#                 })
#                 wandb.log({"LR" : optimizer.param_groups[0]["lr"]})
#                 scheduler.step()
#             else:
#                 wandb.log({
#                     "val_loss" : epoch_loss
#                 })

            if kt < best_kendall_tau:
                best_kendall_tau = kt
                best_epoch = epoch

                torch.save({
                        'epoch': epoch,
                        'model_state_dict': model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'best_loss': kt,
                    }, ckpt_tau_path)
                    
    return model


