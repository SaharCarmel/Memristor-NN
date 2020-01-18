import torch
from torch import autograd
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F

IREAD = 1e-9
MVT = 0.144765
CR = 1 
BB45 = 5.1e-5
CRPROG = 0.48
VPROG = 4.5
VDS = 2 

RES = 10
BATCHSIZE = 256

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                        transform=transforms.Compose([
                            transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))
                        ])),
        batch_size=BATCHSIZE, shuffle=True, **kwargs)


Vth_pos = torch.ones(784,1012) + 0.01*torch.rand((784,1012))
Vth_pos.requires_grad = True

Ids_pos = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos/MVT)
dvthdt_pos = BB45*(CRPROG*VPROG - Vth_pos)
W_pos = (CR/MVT * Ids_pos)

Vth_neg = torch.ones(784,1012) + 0.01*torch.rand((784,1012))
Vth_neg.requires_grad = True

Ids_neg = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg/MVT)
dvthdt_neg = BB45*(CRPROG*VPROG - Vth_neg)
W_neg = (CR/MVT * Ids_neg)


Vth_pos2 = torch.ones(1012,10) + 0.01*torch.rand((1012,10))
Vth_pos2.requires_grad = True

Ids_pos2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos2/MVT)
dvthdt_pos2 = BB45*(CRPROG*VPROG - Vth_pos2)
W_pos2 = (CR/MVT * Ids_pos2)

Vth_neg2 = torch.ones(1012,10) + 0.01*torch.rand((1012,10))
Vth_neg2.requires_grad = True

Ids_neg2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg2/MVT)
dvthdt_neg2 = BB45*(CRPROG*VPROG - Vth_neg2)
W_neg2 = (CR/MVT * Ids_neg2)

smp = list()
criterion =  torch.nn.CrossEntropyLoss(reduction='sum')
for ii in range(10):
    for batch_idx, (data, target) in enumerate(train_loader):
        
        x = data.view((data.shape[0],784))
        x = torch.mm(x,W_pos-W_neg)
        x = F.relu(x)
        x = torch.mm(x,W_pos2-W_neg2)
        output = x
        loss=criterion(output,target)
        
        loss.backward(retain_graph=True)
        

        dvthdt_pos = BB45*(CRPROG*VPROG - Vth_pos)
        with torch.no_grad():
            Vth_pos += RES*torch.mul(dvthdt_pos,F.relu(torch.sign(-Vth_pos.grad)))
            
        Ids_pos = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos/MVT)
        W_pos = (CR/MVT * Ids_pos)


        dvthdt_neg = BB45*(CRPROG*VPROG - Vth_neg)
        with torch.no_grad():
            Vth_neg += RES*torch.mul(dvthdt_neg,F.relu(torch.sign(-Vth_neg.grad)))
            
        Ids_neg = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg/MVT)
        W_neg = (CR/MVT * Ids_neg)


        dvthdt_pos2 = BB45*(CRPROG*VPROG - Vth_pos2)
        with torch.no_grad():
            Vth_pos2 += RES*torch.mul(dvthdt_pos2,F.relu(torch.sign(-Vth_pos2.grad)))
            
        Ids_pos2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos2/MVT)
        W_pos2 = (CR/MVT * Ids_pos2)


        dvthdt_neg2 = BB45*(CRPROG*VPROG - Vth_neg2)
        with torch.no_grad():
            Vth_neg2 += RES*torch.mul(dvthdt_neg2,F.relu(torch.sign(-Vth_neg2.grad)))
            
        Ids_neg2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg2/MVT)
        W_neg2 = (CR/MVT * Ids_neg2)



        pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        print(correct,loss.item(),float(Vth_neg.data[200,5]))
        smp.append(float(Vth_neg.data[200,5]))

print()
    

