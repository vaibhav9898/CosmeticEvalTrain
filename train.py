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

import os
import cv2

import argparse
import torch.optim.lr_scheduler as lr_scheduler

from utils.base_utils import *
from utils.train_utils import train_model_regression
from dataloaders.SymnetDataset import SymnetDatasetNewOne
from dataloaders.SymAug import Augmentation

from networks.models import SymNetNewOne
from losses.losses import SymLoss

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Backbone comparison")
    # Define command-line arguments
    parser.add_argument("--gpu_id", type=str, default='0', help='GPU ID For Train')
    parser.add_argument("--train_file_path", type=str, default="../Data/train_gen.tsv", help="Path to the training file")
    parser.add_argument("--val_file_path", type=str, default="../Data/test.tsv", help="Path to the validation file")
    parser.add_argument("--base_ckpt_path", type=str, default="Checkpoints_GenAI", help="Path to the base checkpoint dir")
    parser.add_argument("--ckpt_tau_path", type=str, default="Tau.pth", help="Path to the tau checkpoint")
    parser.add_argument("--ckpt_accuracy_path", type=str, default="Accuracy.pth", help="Path to the accuracy checkpoint")
    parser.add_argument("--final_model_path", type=str, default="Last.pth", help="Path to the final model checkpoint")
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight Decay")
    parser.add_argument("--nep", type=int, default=200, help="Number of Epochs")
    parser.add_argument("--model_type", type=str, default="resnet", help="Model Backbone")
    parser.add_argument("--image_size", type=int, default="224", help="Image Size : 224 or 256")
    parser.add_argument("--lr_start", type=float, default=0.001, help="Learning Rate Start")
    parser.add_argument("--lr_end", type=float, default=0.000001, help="Learning Rate End")
    parser.add_argument("--scheduler_reqd", type=int, default=1, help="use LR scheduler")
    # Parse command-line arguments
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    train_file_path = args.train_file_path
    val_file_path = args.val_file_path
    base_ckpt_path = args.base_ckpt_path
    
    cuda_device = args.cuda_device
    BATCH_SIZE = args.batch_size
    weight_decay = args.weight_decay
    nep = args.nep
#     kpt_flag = args.kpt           
    model_type = args.model_type
    scheduler_reqd = args.scheduler_reqd
    
    # Here you have three models
    # -> One is model with best Kendall's Tau
    # -> One is model with best Accuracy
    # -> One is Final Model after training
    
    ckpt_tau_path = os.path.join(args.base_ckpt_path, model_type + "_" + args.ckpt_tau_path)
    ckpt_accuracy_path = os.path.join(args.base_ckpt_path, model_type + "_" + args.ckpt_accuracy_path)
    final_model_path = os.path.join(args.base_ckpt_path, model_type + "_" + args.final_model_path)
    
    # Sample Image Size Depending upon Model type we use
    
    if model_type == 'swin' or model_type == 'mobileVIT':
        im_size = 256
    elif model_type == 'resnet' or model_type == 'densenet' or model_type == 'convnext' or model_type == 'convmixer' or model_type == 'eightnet':
        im_size = 224
    else:
        raise("Error : Model out of provided settings")
    
    lr_start = args.lr_start
    lr_end = args.lr_end
    
    print("--train_file_path:", args.train_file_path)
    print("--val_file_path:", args.val_file_path)
    print("--base_ckpt_path:", args.base_ckpt_path)
    print("--ckpt_tau_path:", ckpt_tau_path)
    print("--ckpt_accuracy_path:", ckpt_accuracy_path)
    print("--final_model_path:", final_model_path)
    print("--cuda_device:", args.cuda_device)
    print("--batch_size:", args.batch_size)
    print("--weight_decay:", args.weight_decay)
    print("--nep:", args.nep)
    print("--model_type:", args.model_type)
    print("--image_size:", args.image_size)
    print("--lr_start:", args.lr_start)
    print("--lr_end:", args.lr_end)
    print("--scheduler_reqd:", args.scheduler_reqd)
    

    # Load train and test dataset
    
    train_dataset = SymnetDatasetNewOne(train_file_path, True, im_size=im_size, crop=True)
    val_dataset = SymnetDatasetNewOne(val_file_path, False, im_size=im_size, crop=True)
    
    image_datasets = {
        'train': 
        train_dataset,
        'validation': 
        val_dataset
    }
    
    # Create DataLoaders
    
    dataloaders = {
        'train':
        DataLoader(image_datasets['train'],
                                    batch_size=BATCH_SIZE,
                                    shuffle=True),
        'validation':
        DataLoader(image_datasets['validation'],
                                    batch_size=BATCH_SIZE,
                                    shuffle=False) 
    }
    
    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
  
    #Load model
    model = SymNetNewOne(device=device, model_type=model_type)
    if model_type=="densenet":   
        model.load_state_dict(torch.load("model_checkpoints/densenet_crop_KPT_Concat_Symnet_resnet_generative_data_Accuracy.pth", map_location="cuda:0")['model_state_dict'])
    if model_type=="mobileVIT":   
        model.load_state_dict(torch.load("model_checkpoints/mobileVIT_crop_KPT_Concat_Symnet_resnet_generative_data_Accuracy.pth", map_location="cuda:0")['model_state_dict'])        
    if model_type=="resnet":   
        model.load_state_dict(torch.load("model_checkpoints/resnet_crop_KPT_Concat_Symnet_resnet_generative_data_Accuracy.pth", map_location="cuda:0")['model_state_dict'])
    if model_type=="swin":   
        model.load_state_dict(torch.load("model_checkpoints/swin_crop_KPT_Concat_Symnet_resnet_generative_data_Accuracy.pth", map_location="cuda:0")['model_state_dict'])
        
        
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total trainable parameters: {total_params}")
    
    criterion = SymLoss(mode='Regression', device=device)
    
    optimizer = optim.AdamW(model.parameters(), weight_decay=weight_decay, lr = lr_start)
    
    
    if scheduler_reqd == 1:
        scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=lr_end/lr_start, total_iters=nep)
    else:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, 5, eta_min=0.0002, last_epoch=200)
    
    # Model Training
    model_trained = train_model_regression(model, criterion, optimizer, dataloaders, ckpt_tau_path, ckpt_accuracy_path, nep, device, image_datasets, scheduler)
    
    # Save Model
    torch.save(model_trained.state_dict(), final_model_path)
