import torch
from torch import autograd
from torchvision import datasets, transforms
import numpy as np
import torch.nn.functional as F

from sacred import Experiment
from sacred.utils import apply_backspaces_and_linefeeds
from sacred.observers import MongoObserver


ex = Experiment()
ex.observers.append(MongoObserver.create(db_name='MemristorNN'))
ex.captured_out_filter = apply_backspaces_and_linefeeds



IREAD = 1e-9
MVT = 0.144765
CR = 1 
BB45 = 5.1e-5
CRPROG = 0.48
VPROG = 4.5
VDS = 2 

RES = 10
BATCHSIZE = 256
LS = 800
@ex.automain
def main(_run):

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('../data', train=True, download=True,
                            transform=transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize((0,), (1e-5,))
                            ])),
            batch_size=BATCHSIZE, shuffle=True, **kwargs)


    Vth_pos = torch.ones(784,LS) + 0.1*torch.rand((784,LS))
    Vth_pos.requires_grad = True

    Ids_pos = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos/MVT)
    dvthdt_pos = BB45*(CRPROG*VPROG - Vth_pos)
    W_pos = (CR/MVT * Ids_pos)

    Vth_neg = torch.ones(784,LS) + 0.1*torch.rand((784,LS))
    Vth_neg.requires_grad = True

    Ids_neg = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg/MVT)
    dvthdt_neg = BB45*(CRPROG*VPROG - Vth_neg)
    W_neg = (CR/MVT * Ids_neg)


    Vth_pos2 = torch.ones(LS,10) + 0.1*torch.rand((LS,10))
    Vth_pos2.requires_grad = True

    Ids_pos2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos2/MVT)
    dvthdt_pos2 = BB45*(CRPROG*VPROG - Vth_pos2)
    W_pos2 = (CR/MVT * Ids_pos2)

    Vth_neg2 = torch.ones(LS,10) + 0.1*torch.rand((LS,10))
    Vth_neg2.requires_grad = True

    Ids_neg2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg2/MVT)
    dvthdt_neg2 = BB45*(CRPROG*VPROG - Vth_neg2)
    W_neg2 = (CR/MVT * Ids_neg2)

    smp = list()
    criterion =  torch.nn.CrossEntropyLoss(reduction='sum')
    for ii in range(50):
        for batch_idx, (data, target) in enumerate(train_loader):
            
            x = data.view((data.shape[0],784))
            x = torch.mm(x,W_pos-W_neg)
            # x = F.leaky_relu(x)
            # x = torch.mm(x,W_pos2-W_neg2)
            
            output = x
            loss=criterion(output,target)
            
            loss.backward(retain_graph=True)
            

            dvthdt_pos[dvthdt_pos>0] = BB45*(CRPROG*VPROG - Vth_pos[dvthdt_pos>0])
            dvthdt_pos[-Vth_pos.grad<0] = 0
            with torch.no_grad():
                Vth_pos += RES*dvthdt_pos
                
            Ids_pos = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos/MVT)
            W_pos = (CR/MVT * Ids_pos)


            dvthdt_neg[dvthdt_neg>0] = BB45*(CRPROG*VPROG - Vth_neg[dvthdt_neg>0])
            dvthdt_neg[-Vth_neg.grad<0] = 0
            with torch.no_grad():
                Vth_neg += RES*dvthdt_neg
                
            Ids_neg = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg/MVT)
            W_neg = (CR/MVT * Ids_neg)


            # dvthdt_pos2[dvthdt_pos2>0] = BB45*(CRPROG*VPROG - Vth_pos2[dvthdt_pos2>0])
            # dvthdt_pos2[-Vth_pos2.grad<0] = 0
            # with torch.no_grad():
            #     Vth_pos2 += RES*dvthdt_pos2
                
            # Ids_pos2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_pos2/MVT)
            # W_pos2 = (CR/MVT * Ids_pos2)


            # dvthdt_neg2[dvthdt_neg2>0] = BB45*(CRPROG*VPROG - Vth_neg2[dvthdt_neg2>0])
            # dvthdt_neg2[-Vth_neg2.grad<0] = 0
            # with torch.no_grad():
            #     Vth_neg2 += RES*dvthdt_neg2
                
            # Ids_neg2 = IREAD * np.exp(CR*VDS/MVT) * torch.exp(-Vth_neg2/MVT)
            # W_neg2 = (CR/MVT * Ids_neg2)



            pred = output.argmax(dim=1, keepdim=True) # get the index of the max log-probability
            correct = pred.eq(target.view_as(pred)).sum().item()
            print(correct/data.shape[0]*100,loss.item())

            _run.log_scalar('accuracy',correct/data.shape[0]*100)
            _run.log_scalar('loss',loss.item())
            _run.log_scalar("Vth1",float(Vth_neg[400,50]))
            _run.log_scalar("Vth2",float(Vth_neg[1,50]))
            _run.log_scalar("Vth3",float(Vth_neg[75,9]))
            _run.log_scalar("Vth4",float(Vth_neg[600,100]))
            _run.log_scalar("Vth5",float(Vth_neg[234,5]))
            _run.log_scalar("Vth6",float(Vth_pos[400,50]))
            _run.log_scalar("Vth7",float(Vth_pos[1,50]))
            _run.log_scalar("Vth8",float(Vth_pos[75,9]))
            _run.log_scalar("Vth9",float(Vth_pos[600,100]))
            _run.log_scalar("Vth10",float(Vth_pos[234,5]))
            _run.log_scalar("active_neurons",float(torch.sign(dvthdt_neg).sum()))


    

