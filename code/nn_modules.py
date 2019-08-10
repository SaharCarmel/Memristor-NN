import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class base_net(torch.nn.Module):
    def __init__(self):
        super(base_net, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        #x = F.softmax(self.fc(x),0)
        return x
    
    def update_weights(self,eta):
         self.fc1.weight.data -= eta*self.fc1.weight.grad
         self.fc2.weight.data -= eta*self.fc2.weight.grad

class manhattan_net(torch.nn.Module):
    def __init__(self):
        super(manhattan_net, self).__init__()
        self.fc_pos = nn.Linear(4, 3)
        self.fc_neg = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc_pos(x) - self.fc_neg(x)
        #x = F.softmax(self.fc(x),0)
        return x

    def update_weights(self,eta):
        self.fc_pos.weight.data -= eta*np.sign(self.fc_pos.weight.grad)
        self.fc_neg.weight.data -= eta*np.sign(self.fc_neg.weight.grad)