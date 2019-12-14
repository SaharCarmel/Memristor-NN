import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import models
from nn_modules.Memristor_layer import MemristorLayer


# import math
from torch.nn.parameter import Parameter
import torch.nn.init as init

class ManhattanNetWithModel(torch.nn.Module):
    def __init__(self, args):
        super(ManhattanNetWithModel, self).__init__()
        self.model = models.resnet18(pretrained=True)
        num_ftrs = self.model.fc.in_features
        self.model.fc = nn.Linear(num_ftrs, 120)
        self.fc1_pos = MemristorLayer(120, 60)
        self.fc1_neg = MemristorLayer(120, 60)
        self.fc2_pos = MemristorLayer(60, 10)
        self.fc2_neg = MemristorLayer(60, 10)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.lr = args.lr
        self.smallSignalBias = 0

    def forward(self, x):
        x = self.model(x)
        x = F.relu(self.fc1_pos(x)-self.fc1_neg(x))
        x = self.fc2_pos(x) - self.fc2_neg(x)
        # x = F.softmax(x,0)
        return x

    def update_weights(self):
        self.fc1_pos.update_weight(self.lr)
        self.fc1_neg.update_weight(self.lr)
        self.fc2_pos.update_weight(self.lr)
        self.fc2_neg.update_weight(self.lr)
        

    def optimizer_step(self, epoch):
        self.lr /= (10**epoch)
