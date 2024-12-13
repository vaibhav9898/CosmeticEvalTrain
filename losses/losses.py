import torch
import torch.nn as nn
import numpy as np


class SymLoss(nn.Module):
    def __init__(self, mode, device):
        super(SymLoss, self).__init__()
        self.mode = mode
        if self.mode == 'Classification':
            self.loss = nn.CrossEntropyLoss(weight=torch.tensor([4.720496894409938, 2.4836601307189543, 3.7438423645320196, 8.444444444444445]).to(device))
        elif self.mode == 'Regression':
            self.loss = nn.HuberLoss(reduction = 'mean', delta = 0.001)
#               self.loss = HuberLoss(reduction = 'mean', delta = 0.001)
        else:
            raise ValueError('Class of Error not known')
    def forward(self, output, target):
        # Calculate cross-entropy loss for each output
#         print(output["joint_logits"].shape, target.shape)
        return 1000*self.loss(output["joint_logits"], target)
#         loss1 = self.loss(output["left_logits"], target)
#         loss2 = self.loss(output["right_logits"], target)
#         loss3 = self.loss(output["joint_logits"], target)

#         # Sum the individual losses
#         total_loss = (loss1 + loss2 + loss3)/3.0
#         return total_loss

class HingeLoss(nn.Module):
    def __init__(self, delta_1, delta_2):
        super(HingeLoss, self).__init__()
        self.delta_1 = delta_1
        self.delta_2 = delta_2
        self.relu = nn.ReLU()
    def forward(self, tensor_batch): # tensor_batch has 8 image scores. 0 and 1 are of class 4 and class 1. 2-3-4 are of class 2 and rest from class 3
        l1 = self.relu(tensor_batch[0] + self.delta_1 - tensor_batch[1]) ## c4 + delta_1 <= c1
        l2 = self.relu(tensor_batch[0] + self.delta_2 - tensor_batch[2]) + self.relu(tensor_batch[0] + self.delta_2 - tensor_batch[3]) + self.relu(tensor_batch[0] + self.delta_2 - tensor_batch[4]) 
        l3 = self.relu(tensor_batch[5] + self.delta_2 - tensor_batch[1]) + self.relu(tensor_batch[6] + self.delta_2 - tensor_batch[1]) + self.relu(tensor_batch[7] + self.delta_2 - tensor_batch[1])
#         l1 = max(torch.tensor(0.0), tensor_batch[1] + self.delta_1 - tensor_batch[0])
#         l2 = max(torch.tensor(0.0), tensor_batch[2] + self.delta_2 - tensor_batch[0]) + max(0, tensor_batch[3] + self.delta_2 - tensor_batch[0]) + max(0, tensor_batch[4] + self.delta_2 - tensor_batch[0])
#         l3 = max(torch.tensor(0.0), tensor_batch[1] + self.delta_2 - tensor_batch[5]) + max(0, tensor_batch[1] + self.delta_2 - tensor_batch[6]) + max(0, tensor_batch[1] + self.delta_2 - tensor_batch[7])
        return l1 + l2 + l3





# class HuberLoss(nn.Module):
#     def __init__(self, delta=0.001, reduction='mean'):
#         super(HuberLoss, self).__init__()
#         self.delta = delta
#         self.reduction = reduction

#     def forward(self, input, target):
#         abs_diff = torch.abs(input - target)
#         quadratic = torch.minimum(abs_diff, torch.tensor(self.delta).to(input.device))
#         linear = abs_diff - quadratic
#         loss = 0.5 * quadratic**2 + self.delta * linear

#         if self.reduction == 'mean':
#             return loss.mean()
#         elif self.reduction == 'sum':
#             return loss.sum()
#         else:
#             return loss
