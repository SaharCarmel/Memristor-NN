import argparse
import torch
import yaml
from torchvision import datasets, transforms
import nn_modules
from nn_utils import Args, train, test, digitize_input
import yaml
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver
import time
import os
from torchvision import datasets, models, transforms
# import multiprocessing

# multiprocessing.set_start_method('spawn', True)

_run = 0
args = Args('Parameters.yaml')
net = "manhattan_net_with_model"
digitizeInput = False

use_cuda = not args.noCude and torch.cuda.is_available()

torch.manual_seed(args.seed)

data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ]),
}
# device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
dataloaders = {}
dataloaders['train'] = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=True, download=True,
                          transform=data_transforms['train']),
    batch_size=args.batch_size, shuffle=True, **kwargs)
dataloaders['val'] = torch.utils.data.DataLoader(
    datasets.FashionMNIST('../data', train=False,
                          transform=data_transforms['val']),
    batch_size=args.test_batch_size, shuffle=True, **kwargs)
# data_dir = 'data/hymenoptera_data'
# image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
#                                         data_transforms[x])
#                 for x in ['train', 'val']}
# dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=1,
#                                             shuffle=True, num_workers=0)
#             for x in ['train', 'val']}
# dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
# class_names = image_datasets['train'].classes


if digitizeInput:
    digitize_input(dataloaders['train'], args)
    digitize_input(dataloaders['val'], args)

models = {'ref_net': nn_modules.ref_net, 'manhattan_net': nn_modules.manhattan_net,
          'manhattan_net_with_model': nn_modules.manhattan_net_with_model}
model = models[net](args)
model.to(device)
test_iterator = iter(dataloaders['val'])

start = time.time()

for epoch in range(1, args.epochs):
    train(args, model, device,
          dataloaders['train'], dataloaders['val'],  test_iterator, model.criterion, epoch, _run)
    test(args, model, device, dataloaders['val'], model.criterion, _run)
    model.optimizer_step(epoch)

end = time.time()
_run.log_scalar('Duration', end - start)


if (args.save_model):
    torch.save(model.state_dict(), "mnist_cnn.pt")
