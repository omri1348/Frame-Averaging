

import torch
import torch.nn as nn
from torch.nn import Sequential, Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import (GINConv,global_add_pool)

import numpy as np
from libs.utils import PlanarSATPairsDataset,SpectralDesign
from frame_transform import SortFrame, SampleFrame

MODEL = 'gin' # change to 'mlp' to eval MLP model
FA = True # set false to eval; GA model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform = SpectralDesign(nmax=64,recfield=1,dv=2,nfreq=5,adddegree=True)
dataset = PlanarSATPairsDataset(root="dataset/EXP/",transform=SampleFrame(size=64, sample_size=1, GA=not FA,MLP=MODEL=='mlp'),pre_transform=SortFrame(device, transform))

train_loader = DataLoader(dataset, batch_size=100, shuffle=False)

class GinNet(nn.Module):
    def __init__(self):
        super(GinNet, self).__init__()
        neuron=64
        r1=np.random.uniform()
        r2=np.random.uniform()
        r3=np.random.uniform()

        nn1 = Sequential(Linear(dataset.num_features, neuron))
        self.conv1 = GINConv(nn1,eps=r1,train_eps=True)        

        nn2 = Sequential(Linear(neuron, neuron))
        self.conv2 = GINConv(nn2,eps=r2,train_eps=True)        

        nn3 = Sequential(Linear(neuron, neuron))
        self.conv3 = GINConv(nn3,eps=r3,train_eps=True) 
        
        self.fc1 = torch.nn.Linear(neuron, 10)
        

    def forward(self, data):

        x=data.x 
        edge_index=data.edge_index
        
        x = torch.tanh(self.conv1(x, edge_index))
        x = torch.tanh(self.conv2(x, edge_index))        
        x = torch.tanh(self.conv3(x, edge_index))             

        x = global_add_pool(x, data.batch)
        x = torch.tanh(self.fc1(x))
        return x



class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        input_dim = 4224
        self.conv1 = torch.nn.Linear(input_dim, 2048)
        self.conv2 = torch.nn.Linear(2048, 4096)
        self.conv3 = torch.nn.Linear(4096, 2048) 
        self.fc1 = torch.nn.Linear(2048, 10)
        
    def forward(self, x):
        x = torch.tanh(self.conv1(x))                
        x = torch.tanh(self.conv2(x))        
        x = torch.tanh(self.conv3(x)) 
        x = torch.tanh(self.fc1(x))
        x = x.mean(1)
        return x





M=0
for iter in range(0,100):
    torch.manual_seed(iter)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

    # select your model
    if MODEL == 'mlp':
        model = MlpNet().to(device) 
    else:
        model = GinNet().to(device)

    embeddings=[]
    model.eval()
    for data in train_loader:
        if MODEL == 'mlp':
            x,y = data
            x = x.to(device)
            pre=model(x)
        else:
            data = data.to(device)
            pre=model(data)
        embeddings.append(pre)

    E=torch.cat(embeddings).cpu().detach().numpy()        
    M=M+1*(np.abs(E[0::2]-E[1::2]).sum(1)>0.001)
    sm=(M==0).sum()
    print('similar:',sm)

    