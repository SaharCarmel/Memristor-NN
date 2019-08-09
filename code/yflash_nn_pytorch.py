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
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.datasets import load_iris
import numpy as np

batch_size=10

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    return np.eye(num_classes, dtype='uint8')[y]

iris = load_iris()
x_data=iris.data
y_data=iris.target

#one hot encoding
y_data = to_categorical(y_data,3)

#numpy to pytorch variable
x_data = Variable(torch.from_numpy(x_data))
y_data = Variable(torch.from_numpy(y_data))

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc = nn.Linear(4, 3)

    def forward(self, x):
        x = self.fc(x)
        #x = F.softmax(self.fc(x),0)
        return x


model = Net().double()


#%%



