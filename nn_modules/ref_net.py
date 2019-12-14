import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from torchvision import models


# import math
from torch.nn.parameter import Parameter
import torch.nn.init as init

class RefNet(torch.nn.Module):
    # refernce model see wikipedia MNIST or
    # http://yann.lecun.com/exdb/mnist/
    # 2-layer NN, 800 HU, Cross-Entropy Loss
    def __init__(self, args):
        super(RefNet, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 10)
        self.lr = args.lr
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=0.9)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def update_weights(self):
        self.fc1.weight.data -= self.lr*self.fc1.weight.grad
        self.fc2.weight.data -= self.lr*self.fc2.weight.grad
        # self.optimizer.zero_grad()
        #  self.optimizer.step()

    def optimizer_step(self, epoch):
        pass
