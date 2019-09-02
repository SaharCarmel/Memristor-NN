import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch.optim as optim


# import math
from torch.nn.parameter import Parameter
import torch.nn.init as init


class ref_net(torch.nn.Module):
    # refernce model see wikipedia MNIST or
    # http://yann.lecun.com/exdb/mnist/
    # 2-layer NN, 800 HU, Cross-Entropy Loss
    def __init__(self,args):
        super(ref_net, self).__init__()
        self.fc1 = nn.Linear(784, 800)
        self.fc2 = nn.Linear(800, 10)
        self.lr = args.lr
        self.criterion = nn.CrossEntropyLoss(reduction='sum')   
        self.optimizer = optim.SGD(self.parameters(), lr=args.lr, momentum=0.9)

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
    def update_weights(self):
         self.fc1.weight.data -= self.lr*self.fc1.weight.grad
         self.fc2.weight.data -= self.lr*self.fc2.weight.grad
        # self.optimizer.zero_grad()
        #  self.optimizer.step()
    
    def optimizer_step(self,epoch):
        pass


class manhattan_net(torch.nn.Module):
    def __init__(self,args):
        super(manhattan_net, self).__init__()
        
        self.fc1_pos = Memristor_layer(784, 800)
        self.fc1_neg = Memristor_layer(784, 800)
        self.fc2_pos = Memristor_layer(800, 10)
        self.fc2_neg = Memristor_layer(800, 10)
        
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.lr = args.lr

    def forward(self, x):
        x = x.view(-1,28*28)
        x = F.relu(self.fc1_pos(x)-self.fc1_neg(x))
        x = self.fc2_pos(x) - self.fc2_neg(x)
        #x = F.softmax(self.fc(x),0)
        return x

    def update_weights(self):
        self.fc1_pos.update_weight(self.lr)
        self.fc1_neg.update_weight(self.lr)
        self.fc2_pos.update_weight(self.lr)
        self.fc2_neg.update_weight(self.lr)

    def optimizer_step(self,epoch):
        self.lr /= (10**epoch)



class Memristor_layer(torch.nn.Module):
    __constants__ = ['bias', 'in_features', 'out_features']

    def __init__(self, in_features, out_features, bias=True):
        super(Memristor_layer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=np.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / np.sqrt(fan_in)
            init.uniform_(self.bias, 0, 2*bound)

    def forward(self, input):
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )
    
    def update_weight(self,lr,lower=0,upper=1):
        self.weight.data -= lr*torch.sign(self.weight.grad)
        self.weight.data.requires_grad = False
        self.weight.data[self.weight.data<lower] = lower
        self.weight.data[self.weight.data>upper] = upper
        self.weight.data.requires_grad = True