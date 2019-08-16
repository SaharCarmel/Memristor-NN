from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from torchvision import datasets, transforms
import nn_modules
from nn_utils import Args , train , test


# Training settings
args = Args('Parameters.yml')
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
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)

model = nn_modules.base_net().to(device)
criterion = nn.CrossEntropyLoss(reduction='sum')

for epoch in range(1, args.epochs + 1):
    train(args, model, device, train_loader, criterion , epoch)
    test(args, model, device, test_loader, criterion)

if (args.save_model):
    torch.save(model.state_dict(),"mnist_cnn.pt")