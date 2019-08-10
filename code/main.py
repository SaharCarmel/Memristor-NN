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
os.chdir('code')
import nn_modules
import torchvision.transforms as transforms

#yamlFile = sys.argv[1]

## Checking if cuda available
# use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

yamlFile= 'Parameters.yml'
# def load_param(yamlfile,version,parm_name):
#         version = 'version' + str(version)
#         with open(yamlfile) as f:
#                 data = yaml.load(f, Loader=yaml.FullLoader)
#                 return data[version][parm_name]

## Loading parameter
# batchSize = load_param(yamlFile,1,'batchSize')
batchSize = 10


transform = transforms.Compose(
    [transforms.ToTensor()])
## Loading datasets
trainset = datasets     .MNIST('./data',train=True,transform=transform,download=True)
trainloader = torch.utils.data.DataLoader(trainset,
        batch_size=batchSize, shuffle=True)

testset = datasets.MNIST('./data',train=False,transform=transform,download=True)
testLoader = torch.utils.data.DataLoader(testset,
        batch_size=batchSize, shuffle=True)


model = nn_modules.base_net().float()

#binary cross entropy loss
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

print(trainloader)
train_loss = list()
train_acc = list()
test_loss = list()
test_acc = list()

for epoch in range(2):
    print ("Epoch: "+str(epoch))
    for i , data in enumerate(trainloader,0):
        #print ("batch: "+str(i))
        
        inputs, labels = data
        inputs = inputs.view(-1,28*28)
        print(inputs.shape)
        outputs = model(inputs)

        loss=criterion(outputs,labels)
        #print(epoch, loss.data)
        
        model.zero_grad()
        loss.backward()
        

        model.update_weights(eta=0.005)
                
        train_loss.append(loss.item())

        # out = model(x_train)
#         # _, for epoch in range(2):  # loop over the dataset multiple times


#         # train_acc.append((predicted == y_train).sum().item()/y_train.size(0))
        
#         # out = model(x_test)
#         # test_loss.append(criterion(out,y_test).item())
#         # _, predicted = torch.max(out.data, 1)
#         # test_acc.append((predicted == y_test).sum().item()/y_test.size(0))
        
print('train loss', train_loss[-1])        
# # print('test loss', test_loss[-1])
# # print('train accuracy', train_acc[-1])
# # print('test accuracy', test_acc[-1])

