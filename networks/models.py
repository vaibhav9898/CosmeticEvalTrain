import torch
from torchvision import datasets, models, transforms
import torch.nn as nn
import numpy as np
import torchvision
import os
import cv2
from transformers import ConvNextV2ForImageClassification, Swinv2ForImageClassification, MobileViTV2ForImageClassification
import torch.nn.functional as F

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x

def ConvMixer(dim, depth, inp_channels, kernel_size=9, patch_size=7, n_classes=1):
    return nn.Sequential(
        nn.Conv2d(inp_channels, dim, kernel_size=patch_size, stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
                Residual(nn.Sequential(
                    nn.Conv2d(dim, dim, kernel_size, groups=dim, padding="same"),
                    nn.GELU(),
                    nn.BatchNorm2d(dim)
                )),
                nn.Conv2d(dim, dim, kernel_size=1),
                nn.GELU(),
                nn.BatchNorm2d(dim)
        ) for i in range(depth)],
        nn.AdaptiveAvgPool2d((1,1)),
        nn.Flatten(),
#         nn.Linear(dim, n_classes),
#         nn.Sigmoid()
    )

class SymNetRegKPT(nn.Module):
    def __init__(self, device='cuda:0', model_type = 'resnet', n_class=1):
        super().__init__()
        
        self.device = device
        self.model_type = model_type
        if model_type == 'resnet':
        
            self.single_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
            self.joint_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)

            self.single_feature_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)
            self.joint_feature_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)
            #Similarity based classifier
            # Pass the main, left and right through feature extractor
            # Now pass main through a similarity network so that we can determine the output class
            self.single_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension
            self.joint_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension

            self.single_output = nn.Sequential(
                   nn.Linear(2048,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(2048,n_class), nn.Sigmoid()).to(self.device)
        
        elif model_type == 'swin':
            
            self.single_dchannel = nn.Conv2d(4, 3, kernel_size=(1, 1)).to(self.device)
            self.joint_dchannel = nn.Conv2d(4, 3, kernel_size=(1, 1)).to(self.device)
            
            self.single_feature_extractor_1 = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
            self.joint_feature_extractor_1 = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        
            self.single_feature_extractor_1.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor_1.classifier = nn.Identity().to(self.device) 

            self.single_feature_extractor = nn.Sequential(self.single_dchannel, self.single_feature_extractor_1)
            self.joint_feature_extractor = nn.Sequential(self.joint_dchannel, self.joint_feature_extractor_1)
            
            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'mobileVIT':
            self.single_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
            self.joint_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
            
            self.single_feature_extractor.mobilevitv2.conv_stem.convolution = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(self.device)
            self.joint_feature_extractor.mobilevitv2.conv_stem.convolution = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False).to(self.device)
            
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'convnext':
            
            self.single_dchannel = nn.Conv2d(4, 3, kernel_size=(1, 1)).to(self.device)
            self.joint_dchannel = nn.Conv2d(4, 3, kernel_size=(1, 1)).to(self.device)
            
            self.single_feature_extractor_1 = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
            self.joint_feature_extractor_1 = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
        
            self.single_feature_extractor_1.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor_1.classifier = nn.Identity().to(self.device) 

            self.single_feature_extractor = nn.Sequential(self.single_dchannel, self.single_feature_extractor_1)
            self.joint_feature_extractor = nn.Sequential(self.joint_dchannel, self.joint_feature_extractor_1)
            
            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'densenet':
            self.single_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
            self.joint_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
            
            self.single_feature_extractor.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)
            self.joint_feature_extractor.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)
            
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

            self.single_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'convmixer':
            self.single_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=4).to(self.device)
            self.joint_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=4).to(self.device)
            
#             self.single_feature_extractor.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)
#             self.joint_feature_extractor.features.conv0 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(self.device)
            
#             self.single_feature_extractor.classifier = nn.Identity().to(self.device)
#             self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

#             self.single_output = nn.Sequential(
#                    nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
#             self.joint_output = nn.Sequential(
#                    nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
        
        else:
            raise ValueError('Class of Model not known')
               
    def forward(self, x):
        
        if self.model_type == 'convmixer':
            right_logits = self.single_feature_extractor(x["right"].to(self.device))
            left_logits = self.single_feature_extractor(x["left"].to(self.device))
            joint_logits = self.joint_feature_extractor(x["image"].to(self.device))
            return {
                "left_logits" : left_logits,
                "right_logits" : right_logits,
                "joint_logits" : joint_logits
            }
        
        elif self.model_type == 'resnet' or self.model_type == 'densenet' or self.model_type == 'convmixer':
            right_features = self.single_feature_extractor(x["right"].to(self.device))
            left_features = self.single_feature_extractor(x["left"].to(self.device))
            joint_features = self.joint_feature_extractor(x["image"].to(self.device))

        # For MobViT, Swin and ConvNext
        else:
            right_features = self.single_feature_extractor(x["right"].to(self.device)).logits
            left_features = self.single_feature_extractor(x["left"].to(self.device)).logits
            joint_features = self.joint_feature_extractor(x["image"].to(self.device)).logits

        joint_logits = self.joint_output(joint_features)  #The joint features (from the main input) are passed through a fully connected layer (joint_output) to produce the final joint_logits.
        left_logits = self.single_output(left_features)
        right_logits = self.single_output(right_features)
        return {
            "left" : left_features,
            "right" : right_features,
            "joint" : joint_features,
            "left_logits" : left_logits,
            "right_logits" : right_logits,
            "joint_logits" : joint_logits
        }

class SymNetRegBase(nn.Module):
    def __init__(self, device='cuda:0', model_type = 'resnet', n_class=1):
        super().__init__()
        self.device = device
        self.model_type = model_type
        if model_type == 'resnet':
            self.single_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
            self.right_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
            self.joint_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        
#         self.single_feature_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
#         self.joint_feature_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
        #Similarity based classifier
        # Pass the main, left and right through feature extractor
        # Now pass main through a similarity network so that we can determine the output class
            self.single_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension
            self.right_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension
            self.joint_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension

            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            self.right_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'swin':
            self.single_feature_extractor =Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
            self.joint_feature_extractor = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'mobileVIT':
            self.single_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
            self.joint_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'convnext':
            self.single_feature_extractor = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
            self.joint_feature_extractor = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'densenet':
            self.single_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
            self.joint_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

            self.single_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'convmixer':
            self.single_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=3).to(self.device)
            self.joint_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=3).to(self.device)
        else:
            raise ValueError('Class of Model not known')
               
    def forward(self, x):
        
        if self.model_type == 'convmixer':
            right_logits = self.single_feature_extractor(x["right"].to(self.device))
            left_logits = self.single_feature_extractor(x["left"].to(self.device))
            joint_logits = self.joint_feature_extractor(x["image"].to(self.device))
            return {
                "left_logits" : left_logits,
                "right_logits" : right_logits,
                "joint_logits" : joint_logits
            }
            

        # For DenseNet and ResNet
        elif self.model_type == 'resnet' or self.model_type == 'densenet':
            right_features = self.right_feature_extractor(x["right"].to(self.device))
            left_features = self.single_feature_extractor(x["left"].to(self.device))
            joint_features = self.joint_feature_extractor(x["image"].to(self.device))

        # For MobViT, Swin and ConvNext
        else:
            right_features = self.single_feature_extractor(x["right"].to(self.device)).logits
            left_features = self.single_feature_extractor(x["left"].to(self.device)).logits
            joint_features = self.joint_feature_extractor(x["image"].to(self.device)).logits

    
        joint_logits = self.joint_output(joint_features)
        left_logits = self.single_output(left_features)
        right_logits = self.right_output(right_features)
        return {
            "left" : left_features,
            "right" : right_features,
            "joint" : joint_features,
            "left_logits" : left_logits,
            "right_logits" : right_logits,
            "joint_logits" : joint_logits
        }

class SymNetClassification(nn.Module):
    def __init__(self, device, n_class=4):
        super().__init__()
        self.device = device
        self.single_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        self.joint_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
        
        #Similarity based classifier
        # Pass the main, left and right through feature extractor
        # Now pass main through a similarity network so that we can determine the output class
        self.single_feature_extractor.fc = nn.Sequential(
               nn.Linear(2048,n_class)).to(self.device) # 2048 dimension
        self.joint_feature_extractor.fc = nn.Sequential(
               nn.Linear(2048,n_class)).to(self.device)
  
    def forward(self, x):
        right_logits = self.single_feature_extractor(x["right"].to(self.device))
        left_logits = self.single_feature_extractor(x["left"].to(self.device))
        joint_logits = self.joint_feature_extractor(x["image"].to(self.device))
    
        return {
            "left_logits" : left_logits,
            "right_logits" : right_logits,
            "joint_logits" : joint_logits
        }    

class EightNet(nn.Module): #base class for neural networks in Pytorch
    def __init__(self, device, model_type='resnet', final_model_path = None): #allows loading pre-trained weights from a file if a path is provided
        super(EightNet, self).__init__()
        self.device = device
        self.feature_extractor = SymNetNewOne(device=device, model_type = 'resnet')
        if final_model_path is not None:    
            self.feature_extractor.load_state_dict(torch.load(final_model_path, map_location="cuda:0")["model_state_dict"])
  
    def forward(self, x):
        return self.feature_extractor(x)

class BaseNetClass(nn.Module):
    def __init__(self, device='cuda:0', model_type = 'resnet', n_class=4):
        super().__init__()
        self.device = device
        self.model_type = model_type
        if model_type == 'resnet':
            self.single_feature_extractor = models.resnet50(weights=models.ResNet50_Weights.DEFAULT).to(self.device)
            self.single_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension
            self.single_output = nn.Sequential(
                   nn.Linear(2048,n_class)).to(self.device)
        elif model_type == 'swin':
            self.single_feature_extractor =Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)

            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class)).to(self.device)
        elif model_type == 'mobileVIT':
            self.single_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class)).to(self.device)
            
        elif model_type == 'convnext':
            self.single_feature_extractor = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class)).to(self.device)
        elif model_type == 'densenet':
            self.single_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.single_output = nn.Sequential(
                   nn.Linear(1920,n_class)).to(self.device)
            
        elif model_type == 'convmixer':
            self.single_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=3).to(self.device)
        else:
            raise ValueError('Class of Model not known')
               
    def forward(self, x):
        
        if self.model_type == 'convmixer':
            joint_logits = self.single_feature_extractor(x["image"].to(self.device))
            return {
                "joint_logits" : right_logits
            }
        # For DenseNet and ResNet
        elif self.model_type == 'resnet' or self.model_type == 'densenet':
            joint_features = self.single_feature_extractor(x["image"].to(self.device))

        # For MobViT, Swin and ConvNext
        else:
            joint_features = self.single_feature_extractor(x["image"].to(self.device)).logits
    
        joint_logits = self.single_output(joint_features)
        return {
            "joint_logits" : joint_logits
        }

    

class BaseNetClassRegression(nn.Module):
    def __init__(self, device='cuda:0', model_type = 'resnet', n_class=1):
        super().__init__()
        self.device = device
        self.model_type = model_type
        if model_type == 'resnet':
            self.single_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
            self.single_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension
            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'swin':
            self.single_feature_extractor =Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)

            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'mobileVIT':
            self.single_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            
        elif model_type == 'convnext':
            self.single_feature_extractor = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'densenet':
            self.single_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.single_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
            
        elif model_type == 'convmixer':
            self.single_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=3).to(self.device)
        else:
            raise ValueError('Class of Model not known')
               
    def forward(self, x):
        
        if self.model_type == 'convmixer':
            joint_logits = self.single_feature_extractor(x["image"].to(self.device))
            return {
                "joint_logits" : right_logits
            }
        # For DenseNet and ResNet
        elif self.model_type == 'resnet' or self.model_type == 'densenet':
            joint_features = self.single_feature_extractor(x["image"].to(self.device))

        # For MobViT, Swin and ConvNext
        else:
            joint_features = self.single_feature_extractor(x["image"].to(self.device)).logits
    
        joint_logits = self.single_output(joint_features)
        return {
            "joint_logits" : joint_logits
        }

class FFNN(nn.Module):
    def __init__(self, input_size, hidden_size_1, output_size):
        super(FFNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size_1)
        self.fc2 = nn.Linear(hidden_size_1, output_size)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = nn.Sigmoid()(self.fc2(x))
        return x


class SymNetNewOne(nn.Module):
    def __init__(self, device='cuda:0', model_type = 'resnet', n_class=1):
        super().__init__()
        self.device = device
        self.model_type = model_type
        if model_type == 'resnet':
            self.single_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device) #Loads a pre-trained ResNet-18 model using the default weights for the "single" input and moves it to the specified device (GPU or CPU).

            self.right_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
            self.joint_feature_extractor = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
            
#             self.single_feature_extractor = ResNet18Custom(dropout_prob=0.5).to(self.device).to(self.device)
#             self.right_feature_extractor = ResNet18Custom(dropout_prob=0.5).to(self.device).to(self.device)
#             self.joint_feature_extractor = ResNet18Custom(dropout_prob=0.5).to(self.device).to(self.device)
            
            
            
        
#         self.single_feature_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
#         self.joint_feature_extractor.conv1 = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False).to(device)
        #Similarity based classifier
        # Pass the main, left and right through feature extractor
        # Now pass main through a similarity network so that we can determine the output class
        
            self.single_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension 
            # This removes the classification layer since we require the feature vectors and not classification

            self.right_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension
            self.joint_feature_extractor.fc = nn.Identity().to(self.device) # 2048 dimension
            
            #Uncomment ffnn_1 = FFNN for old results
            self.ffnn_1 = FFNN(512*3 + 76, 512, 1).to(self.device) #512 is feature vector size and 3 is ResNet Models and 76 is KeyPoints
            
            #Uncomment ffnn_1 = NET for new results
#             hidden = [50,50,0]
#             dropout = 0.1
#             # Define the model on the GPU
#             self.ffnn_1 = Net(512*3 + 76, hidden, 1, dropout).to(self.device)
            
            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            self.right_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'swin':
            self.single_feature_extractor = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
            self.right_feature_extractor = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
            self.joint_feature_extractor = Swinv2ForImageClassification.from_pretrained("microsoft/swinv2-tiny-patch4-window8-256").to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.right_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 
            
            self.ffnn_1 = FFNN(768*3 + 76, 512, 1).to(self.device)
            
            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.right_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'mobileVIT':
            self.single_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
            self.right_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
            self.joint_feature_extractor = MobileViTV2ForImageClassification.from_pretrained("apple/mobilevitv2-1.0-imagenet1k-256").to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.right_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 
            
            self.ffnn_1 = FFNN(512*3 + 76, 512, 1).to(self.device)
            self.single_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
            self.right_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(512,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'convnext':
            self.single_feature_extractor = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
            self.right_feature_extractor = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
            self.joint_feature_extractor = ConvNextV2ForImageClassification.from_pretrained("facebook/convnextv2-tiny-1k-224").to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.right_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 
            
            self.ffnn_1 = FFNN(768*3 + 76, 512, 1).to(self.device)
            
            self.single_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.right_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(768,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'densenet':
            self.single_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
            self.right_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
            self.joint_feature_extractor = models.densenet201(weights=models.DenseNet201_Weights.DEFAULT).to(self.device)
        
            self.single_feature_extractor.classifier = nn.Identity().to(self.device)
            self.right_feature_extractor.classifier = nn.Identity().to(self.device)
            self.joint_feature_extractor.classifier = nn.Identity().to(self.device) 

            self.ffnn_1 = FFNN(1920*3 + 76, 512, 1).to(self.device)
            
            self.single_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
            self.right_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
            self.joint_output = nn.Sequential(
                   nn.Linear(1920,n_class), nn.Sigmoid()).to(self.device)
        elif model_type == 'convmixer':
            self.single_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=3).to(self.device)
            self.right_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=3).to(self.device)
            self.joint_feature_extractor = ConvMixer(dim=768, depth=32, inp_channels=3).to(self.device)
            
            self.ffnn_1 = FFNN(768*3 + 76, 512, 1).to(self.device)
        else:
            raise ValueError('Class of Model not known')
               
    def forward(self, x):
        
        if self.model_type == 'convmixer':
            right_features = self.right_feature_extractor(x["right"].to(self.device))
            left_features = self.single_feature_extractor(x["left"].to(self.device))
            joint_features = self.joint_feature_extractor(x["image"].to(self.device))
#             return {
#                 "left_logits" : left_logits,
#                 "right_logits" : right_logits,
#                 "joint_logits" : joint_logits
#             }
            

        # For DenseNet and ResNet
        elif self.model_type == 'resnet' or self.model_type == 'densenet':
            right_features = self.right_feature_extractor(x["right"].to(self.device))
            left_features = self.single_feature_extractor(x["left"].to(self.device))
            joint_features = self.joint_feature_extractor(x["image"].to(self.device))
            
            # print(right_features.shape, left_features.shape, joint_features.shape, x["flattened_landmarks"].shape)
#             concatenated_tensor = torch.cat((right_features, left_features, joint_features, x["flattened_landmarks"].to("cuda")), dim=1)
#             fin_op = self.ffnn_1(concatenated_tensor)
#             return {
#                 "joint_logits" : fin_op
#             }
            
            
            
        # For MobViT, Swin and ConvNext
        else:
            right_features = self.right_feature_extractor(x["right"].to(self.device)).logits
            left_features = self.single_feature_extractor(x["left"].to(self.device)).logits
            joint_features = self.joint_feature_extractor(x["image"].to(self.device)).logits
        
        concatenated_tensor = torch.cat((right_features, left_features, joint_features, x["flattened_landmarks"].to("cuda")), dim=1)
        fin_op = self.ffnn_1(concatenated_tensor)
        return {
            "joint_logits" : fin_op
        }

#         joint_logits = self.joint_output(joint_features)
#         left_logits = self.single_output(left_features)
#         right_logits = self.right_output(right_features)
#         return {
#             "left" : left_features,
#             "right" : right_features,
#             "joint" : joint_features,
#             "left_logits" : left_logits,
#             "right_logits" : right_logits,
#             "joint_logits" : joint_logits
#         }

class Net(nn.Module):
    def __init__(self, in_dim, hidden_dims, out_dim, dropout):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        #self.fc3 = nn.Linear(hidden_dims[1], hidden_dims[2]) # Third hidden layer not used
        self.fc4 = nn.Linear(hidden_dims[1], out_dim)
        self.dropout1 = nn.Dropout(p=dropout)
        self.dropout2 = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout1(F.elu(self.fc1(x)))
        x = self.dropout2(F.elu(self.fc2(x)))
        #x = self.dropout2(F.elu(self.fc3(x))) # Third layer not used
        x = self.fc4(x)
        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, dropout_prob=0.0):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.bn2(self.conv2(out))
        out = self.dropout(out)
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNetCustom(nn.Module):
    def __init__(self, block, num_blocks, num_classes=1000, dropout_prob=0.0):
        super(ResNetCustom, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1, dropout_prob=dropout_prob)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2, dropout_prob=dropout_prob)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2, dropout_prob=dropout_prob)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2, dropout_prob=dropout_prob)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.expansion, num_classes)

        self.dropout = nn.Dropout(dropout_prob)

    def _make_layer(self, block, planes, num_blocks, stride, dropout_prob):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, dropout_prob))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

def ResNet18Custom(dropout_prob=0.0):
    print("Custom Initialization")
    return ResNetCustom(BasicBlock, [2, 2, 2, 2], dropout_prob=dropout_prob)

# Example usage:
# model = ResNet18Custom(dropout_prob=0.5)
