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
