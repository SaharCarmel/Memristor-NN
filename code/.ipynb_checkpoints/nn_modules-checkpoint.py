import torch
import torch.nn as nn
import torch.nn.functional as F

class base_net(torch.nn.Module):
    def __init__(self):
        super(base_net, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc(x)
        #x = F.softmax(self.fc(x),0)
        return x

class manhattan_net(torch.nn.Module):
    def __init__(self):
        super(manhattan_net, self).__init__()
        self.fc_pos = nn.Linear(4, 3)
        self.fc_neg = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc_pos(x) - self.fc_neg(x)
        #x = F.softmax(self.fc(x),0)
        return x