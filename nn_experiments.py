import argparse
import torch
import yaml
from torchvision import datasets, transforms
import nn_modules
from nn_utils import Args , train , test , digitize_input
import yaml
from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver
import time




ex = Experiment()
ex.observers.append(MongoObserver.create(db_name='MemristorNN'))
ex.captured_out_filter = apply_backspaces_and_linefeeds


# Training settings
@ex.config
def my_config():
    args = Args('Parameters.yaml')
    net = "ref_net"
    digitizeInput = False
    lower = 0
    upper = 1
    args.lower = lower
    args.upper = upper


@ex.automain
def main(args,digitizeInput,net,_run):
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

    if digitizeInput:
        digitize_input(train_loader,args)
        digitize_input(test_loader,args)
    
    models = {'ref_net' : nn_modules.ref_net , 'manhattan_net' : nn_modules.manhattan_net}
    model = models[net](args)
    model.to(device) 
    test_iterator = iter(test_loader)

    start = time.time()
    
    for epoch in range(1, args.epochs):    
        train(args, model, device, train_loader, test_loader ,  test_iterator, model.criterion , epoch , _run)
        test(args, model, device, test_loader, model.criterion , _run)
        model.optimizer_step(epoch)

    end = time.time()
    _run.log_scalar('Duration',end - start)

    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

# if __name__ == '__main__':
#     main()