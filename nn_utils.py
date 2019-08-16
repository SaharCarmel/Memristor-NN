from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torchvision import datasets, transforms

class Args():
    def __init__(self, yamlfile):
        self.batch_size = self.load_param(yamlfile,"batch_size")
        self.test_batch_size = self.load_param(yamlfile,"test_batch_size")
        self.epochs = self.load_param(yamlfile,"epochs")
        self.lr = self.load_param(yamlfile,"lr")
        self.momentum = self.load_param(yamlfile,"momentum")
        self.noCude = self.load_param(yamlfile, "noCuda")
        self.seed = self.load_param(yamlfile,"seed")
        self.log_interval = self.load_param(yamlfile,"log_interval")
        self.save_model = self.load_param(yamlfile,"save_model")

    
    def load_param(self,yamlfile,parm_name):
        with open(yamlfile) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                return data[parm_name]
        pass



def train(args, model, device, train_loader, criterion ,epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss=criterion(output,target)

        model.zero_grad()
        loss.backward()
        model.update_weights(args.lr)

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def test(args, model, device, test_loader,criterion):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))