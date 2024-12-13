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

from utils.base_utils import *
from dataloaders.SymnetDataset import SymnetDatasetNewOneTest
from dataloaders.SymAug import Augmentation
from networks.models import SymNetNewOne
from losses.losses import SymLoss
from utils.test_utils import test_combined_model_regression


import argparse
import torch.optim.lr_scheduler as lr_scheduler


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference")

#     # Define command-line arguments
#     parser.add_argument("--val_file_path", type=str, default="../Data/dataset/annot_dir/TMH/test.tsv", help="Path to the validation file")
#     parser.add_argument("--gpu_id", type=str, default="0", help="GPU ID")
#     parser.add_argument("--batch_size", type=int, default=4, help="Batch size")

    # Path to the validation file
    parser.add_argument(
        "--val_file_path",
        type=str,
        default="../Data/dataset/annot_dir/TMH/test.tsv",
        help="Path to the validation file"
    )

    # GPU ID
    parser.add_argument(
        "--gpu_id",
        type=str,
        default="0",
        help="GPU ID to use (e.g., 'cuda:0')"
    )

    # Batch size
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Batch size for processing"
    )

    # Model checkpoint paths
    parser.add_argument(
        "--densenet_path",
        type=str,
        required=True,
        help="Path to the DenseNet model checkpoint"
    )

    parser.add_argument(
        "--resnet_path",
        type=str,
        required=True,
        help="Path to the ResNet model checkpoint"
    )

    parser.add_argument(
        "--mob_path",
        type=str,
        required=True,
        help="Path to the MobileVIT model checkpoint"
    )

    parser.add_argument(
        "--swin_path",
        type=str,
        required=True,
        help="Path to the Swin Transformer model checkpoint"
    )

    # Parse command-line arguments
    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    val_file_path = args.val_file_path
    BATCH_SIZE = args.batch_size
    
    # Load Test Data for 224 and 256 Images
    
    val_dataset_224 = SymnetDatasetNewOneTest(val_file_path, False, im_size=224, crop=True)
    val_dataset_256 = SymnetDatasetNewOneTest(val_file_path, False, im_size=256, crop=True)
    
    image_datasets_224 = {
        'validation': 
        val_dataset_224
    }

    image_datasets_256 = {
        'validation': 
        val_dataset_256
    }
    
    dataloaders_224 = {
        'validation':
        DataLoader(image_datasets_224['validation'],
                                    batch_size=BATCH_SIZE,
                                    shuffle=False) 
    }
    
    dataloaders_256 = {
        'validation':
        DataLoader(image_datasets_256['validation'],
                                    batch_size=BATCH_SIZE,
                                    shuffle=False) 
    }
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    # Load Models for all the voters
    
    model_1 = SymNetNewOne(device=device, model_type='resnet')
    model_2 = SymNetNewOne(device=device, model_type='swin')
    model_3 = SymNetNewOne(device=device, model_type='densenet')
    model_4 = SymNetNewOne(device=device, model_type='mobileVIT')
    
    
    densenet_path = args.densenet_path
    resnet_path = args.resnet_path
    mob_path = args.mob_path
    swin_path = args.swin_path
    
    model_1.load_state_dict(torch.load(resnet_path, map_location="cuda:0")["model_state_dict"])
    model_1.eval()
    
    model_2.load_state_dict(torch.load(swin_path, map_location="cuda:0")["model_state_dict"])
    model_2.eval()
    
    model_3.load_state_dict(torch.load(densenet_path, map_location="cuda:0")["model_state_dict"])
    model_3.eval()
    
    model_4.load_state_dict(torch.load(mob_path, map_location="cuda:0")["model_state_dict"])
    model_4.eval()
    
    print("Model Loaded")
    
    criterion = SymLoss(mode='Regression', device=device)
    test_combined_model_regression(model_1, model_2, model_3, model_4, dataloaders_224, dataloaders_256, device, image_datasets_224, criterion)

        
