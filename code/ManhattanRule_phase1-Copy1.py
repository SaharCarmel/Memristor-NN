# To add a new cell, type '#%%'
# To add a new markdown cell, type '#%% [markdown]'
#%% Change working directory from the workspace root to the ipynb file location. Turn this addition off with the DataScience.changeDirOnImportExport setting
# ms-python.python added
import os
try:
	os.chdir(os.path.join(os.getcwd(), 'code'))
	print(os.getcwd())
except:
	pass
#%%
from IPython import get_ipython


#%%
import torch
import numpy as np
from torch.autograd import Variable
from sklearn.datasets import load_iris
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from matplotlib import pyplot as plt
import nn_modules

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

#batch size
batch_size=10

#loading iris data from sklearn
iris = load_iris()
r = torch.randperm(len(iris.data))
p = int(np.ceil(len(iris.data)*0.1))

x_train=iris.data[r[p:]]
x_test=iris.data[r[:p]]
y_train=iris.target[r[p:]]
y_test=iris.target[r[:p]]

#one hot encoding
#y_data = to_categorical(y_data,3)

#numpy to pytorch variable
x_train = Variable(torch.from_numpy(x_train)).double()
y_train = Variable(torch.from_numpy(y_train)).long()
x_test = Variable(torch.from_numpy(x_test)).double()
y_test = Variable(torch.from_numpy(y_test)).long()


        

model = nn_modules.base_net().double()

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
        model.fc.weight.data -= 0.005*np.sign(model.fc.weight.grad)
        
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


#%%
get_ipython().run_line_magic('matplotlib', '')
from scipy import signal
N=13
plt.plot(signal.filtfilt(np.ones([N])/N,1,train_acc))
plt.plot(signal.filtfilt(np.ones([N])/N,1,test_acc))
plt.title('Accuracy')
plt.legend({'train','test'})
plt.show()


#%%
get_ipython().run_line_magic('matplotlib', '')
from scipy import signal
N=31
plt.plot(signal.filtfilt(np.ones([N])/N,1,train_loss))
plt.plot(signal.filtfilt(np.ones([N])/N,1,test_loss))
plt.title('Loss')
plt.legend({'train','test'})
plt.show()


#%%
y_test


#%%
y_train


#%%



