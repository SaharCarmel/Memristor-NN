import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import models


# import math
from torch.nn.parameter import Parameter
import torch.nn.init as init

class ManhattanNet(torch.nn.Module):
    def __init__(self, args):
        super(manhattan_net, self).__init__()

        self.fc1_pos = Memristor_layer(784, 800)
        self.fc1_neg = Memristor_layer(784, 800)
        self.fc2_pos = Memristor_layer(800, 2)
        self.fc2_neg = Memristor_layer(800, 2)

        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.lr = args.lr

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1_pos(x)-self.fc1_neg(x))
        x = self.fc2_pos(x) - self.fc2_neg(x)
        #x = F.softmax(self.fc(x),0)
        return x

    def update_weights(self):
        self.fc1_pos.update_weight(self.lr)
        self.fc1_neg.update_weight(self.lr)
        self.fc2_pos.update_weight(self.lr)
        self.fc2_neg.update_weight(self.lr)

    def optimizer_step(self, epoch):
        self.lr /= (10**epoch)
