from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

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
        self.dg_bins = self.load_param(yamlfile,"dg_bins")
        self.dg_values = self.load_param(yamlfile,"dg_values")
        self.bias_voltage = self.load_param(yamlfile, "bias_voltage")

    
    def load_param(self,yamlfile,parm_name):
        with open(yamlfile) as f:
                data = yaml.load(f, Loader=yaml.FullLoader)
                return data[parm_name]
        pass
    
    def __repr__(self):
        print("Batch size: " + str(self.batch_size))
        print("Test batch size: " + str(self.test_batch_size))
        print("Epochs: " + str(self.epochs))
        print("Learning rate: " + str(self.lr))
        print("Momentum: " + str(self.momentum))
        print("No cude: " + str(self.noCude))
        print("Log interval: " + str(self.log_interval))
        print("Save model? " + str(self.save_model))
        print("dg_bins " + str(self.dg_bins))
        print("dg_values " + str(self.dg_values))
        print("bias_voltage " + str(self.bias_voltage))
        return ''


def train(args, model, device, train_loader, test_loader ,  test_iterator, criterion ,epoch , _run):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        output = model(data)
        
        loss=criterion(output,target)

        model.zero_grad()
        loss.backward()
        model.update_weights()

        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

            model.eval()
            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            _run.log_scalar("Train Loss",loss.item())
            _run.log_scalar("Train Accuracy",correct)
            with torch.no_grad():
                try:
                    data, target = test_iterator.next()
                except:
                    test_iterator = iter(test_loader)
                    data, target = test_iterator.next()

                data, target = data.to(device), target.to(device)
                output = model(data)
                test_loss = criterion(output, target).item() # sum up batch loss
                pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
                correct = pred.eq(target.view_as(pred)).sum().item()

                _run.log_scalar("Test Loss",test_loss)
                _run.log_scalar("Test Accuracy (run)",correct)    
            
            



def test(args, model, device, test_loader,criterion, _run):
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
    _run.log_scalar("Test Accuracy",100. * correct / len(test_loader.dataset))

def digitize_input(data_loader,args):
    d = np.digitize(data_loader.dataset.data.numpy(),args.dg_bins)
    for i in range(1,len(args.dg_bins)):
        data_loader.dataset.data[torch.Tensor((d==i).astype(int)).type(torch.ByteTensor)] = args.dg_values[i-1]

def visualize_digitization(args):
    use_cuda = not args.noCude and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    
    plt.figure()
    
    plt.subplot(1,2,1)
    plt.hist(train_loader.dataset.data.numpy().reshape(-1))
    plt.title('DataSet Histogram')

    d = np.digitize(train_loader.dataset.data.numpy(),args.dg_bins)
    for i in range(1,len(args.dg_bins)):
        train_loader.dataset.data[torch.Tensor((d==i).astype(int)).type(torch.ByteTensor)] = args.dg_values[i-1]
    
    plt.subplot(1,2,2)
    plt.hist(train_loader.dataset.data.numpy().reshape(-1))
    plt.title('Digitized DataSet Histogram')

    plt.show()
        
        

