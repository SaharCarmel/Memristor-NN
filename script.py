import argparse
import torch
import yaml
from torchvision import datasets, transforms
import nn_modules
from nn_utils import Args , train , test 
from sacred import Experiment
from sacred.observers import MongoObserver
from sacred.utils import apply_backspaces_and_linefeeds


ex = Experiment()
# ex.observers.append(MongoObserver.create())
ex.captured_out_filter = apply_backspaces_and_linefeeds


# Training settings
@ex.config
def my_config():
    args = Args('Parameters.yml')
    net = "ref_net"


@ex.automain
def main(args):
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

    model = nn_modules.manhattan_net().to(device)
    
    for epoch in range(1, args.epochs + 1):
        args.lr /= (10**(epoch-1))
        train(args, model, device, train_loader, model.criterion , epoch)
        test(args, model, device, test_loader, model.criterion)


    if (args.save_model):
        torch.save(model.state_dict(),"mnist_cnn.pt")

if __name__ == '__main__':
    main()