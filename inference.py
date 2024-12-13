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
from dataloaders.SymnetDataset import SymnetDatasetKeypoint
from dataloaders.SymAug import Augmentation
from networks.models import SymNetRegKPT
from losses.losses import SymLoss
from utils.test_utils import test_model_regression

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")

    # Define command-line arguments
    parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
    parser.add_argument("--val_file_path", type=str, default="../Data/dataset/annot_dir/TMH/test.tsv", help="Path to the validation file")
    parser.add_argument("--final_model_path", type=str, default="Last.pth", help="Path to the final model checkpoint")
    parser.add_argument("--cuda_device", type=str, default="cuda:0", help="CUDA device")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--kpt", type=int, default=0, help="Keypoint Channel")
    parser.add_argument("--model_type", type=str, default="resnet", help="Model Backbone")
    parser.add_argument("--image_size", type=int, default="224", help="Image Size : 224 or 256")

    # Parse command-line arguments
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    
    val_file_path = args.val_file_path
    cuda_device = args.cuda_device
    BATCH_SIZE = args.batch_size
    model_type = args.model_type
    
    final_model_path = args.final_model_path
    
    if model_type == 'swin' or model_type == 'mobileVIT':
        im_size = 256
    elif model_type == 'resnet' or model_type == 'densenet' or model_type == 'convnext' or model_type == 'eightnet' or model_type == 'convmixer':
        im_size = 224
    else:
        raise("Error : Model out of provided settings")
    
    print(f"Validation File Path: {val_file_path}")
    print(f"Final Model Path: {final_model_path}")
    print(f"CUDA Device: {cuda_device}")
    print(f"Batch Size: {BATCH_SIZE}")
    print(f"model_type: {model_type}")
    
    val_dataset = SymnetDatasetKeypoint(val_file_path, False, im_size=im_size)

    image_datasets = {
        'validation': 
        val_dataset
    }

    dataloaders = {
        'validation':
        DataLoader(image_datasets['validation'],
                                    batch_size=BATCH_SIZE,
                                    shuffle=False) 
    }

    device = torch.device(cuda_device if torch.cuda.is_available() else "cpu")
        
    model = SymNetRegKPT(device=device, model_type=model_type)

    #### Load Checkpoint Model here
    model.load_state_dict(torch.load(final_model_path, map_location="cuda:0")["model_state_dict"])
    model.eval()
    
    print("Model Loaded")
    
    criterion = SymLoss(mode='Regression', device=device)
    test_model_regression(model, dataloaders, device, image_datasets, criterion)

        
