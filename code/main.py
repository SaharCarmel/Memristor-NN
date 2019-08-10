import torch
import sys
import os
import yaml
import numpy as np
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import torchvision.datasets as datasets
import nn_modules

#yamlFile = sys.argv[1]

##Validating version
if yaml.__version__ != '5.1.1':
        print('Yaml module version outdated. Please update to module 5.1.1')
        quit()

        
def plotModelOverview(test_acc,train_acc,):
        pass


## Checking if cuda available
use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

yamlFile= 'Parameters.yml'
def load_param(yamlfile,version,parm_name):
        version = 'version' + str(version)
        with open(yamlfile) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                return data[version][parm_name]

## Loading parameter
batchSize = load_param(yamlFile,1,'batchSize')

## Loading datasets
trainset = datasets.MNIST('./data',train=True,transform=None,download=True)
train_loader = torch.utils.data.DataLoader(trainset,
        batch_size=batchSize, shuffle=True)

testset = datasets.MNIST('./data',train=False,transform=None,download=True)
testLoader = torch.utils.data.DataLoader(testset,
        batch_size=batchSize, shuffle=True)


model = nn_modules.manhattan_net().double()

#binary cross entropy loss
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

permutation = torch.randperm(x_train.size()[0])

train_loss = list()
train_acc = list()
test_loss = list()
test_acc = list()
#training
for epoch in range(100):
    #print ("Epoch: "+str(epoch))
    for i in range(0,x_train.size()[0],batch_size):
        #print ("batch: "+str(i))
        indices = permutation[i:i+batch_size]
        inputs, labels = x_train[indices], y_train[indices]
        outputs = model(inputs)

        loss=criterion(outputs,labels)
        #print(epoch, loss.data)
        
        model.zero_grad()
        loss.backward()
        # model.fc_pos.weight.data -= 0.005*np.sign(model.fc_pos.weight.grad)
        # model.fc_neg.weight.data -= 0.005*np.sign(model.fc_neg.weight.grad)

        model.update_weights()
                
        train_loss.append(loss.item())

        out = model(x_train)
        _, predicted = torch.max(out.data, 1)
        train_acc.append((predicted == y_train).sum().item()/y_train.size(0))
        
        out = model(x_test)
        test_loss.append(criterion(out,y_test).item())
        _, predicted = torch.max(out.data, 1)
        test_acc.append((predicted == y_test).sum().item()/y_test.size(0))
        
print('train loss', train_loss[-1])        
print('test loss', test_loss[-1])
print('train accuracy', train_acc[-1])
print('test accuracy', test_acc[-1])

