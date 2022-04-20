
import torch
import torch.nn as nn

from torch.nn import Sequential, Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import (global_add_pool,GINConv)
import numpy as np
from frame_transform import SortFrame, SampleFrame8C


from libs.utils import Grapg8cDataset,SpectralDesign
MODEL = 'gin' # change to 'mlp' to eval MLP model
FA = True # set false to eval; GA model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform = SpectralDesign(nmax=8,recfield=1,dv=2,nfreq=5,adddegree=True)
dataset = Grapg8cDataset(root="dataset/graph8c/",transform=SampleFrame8C(size=8, sample_size=1, GA=not FA,MLP=MODEL=='mlp'),pre_transform=SortFrame(device, transform))

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
        input_dim = 64
        self.conv1 = torch.nn.Linear(input_dim, 128)
        self.conv2 = torch.nn.Linear(128, 64)
        self.fc1 = torch.nn.Linear(64, 10)
        
    def forward(self, x):

        x = torch.tanh(self.conv1(x))                
        x = torch.tanh(self.conv2(x))        
        x = x.mean(1)
        x = torch.tanh(self.fc1(x))
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
    M=M+1*((np.abs(np.expand_dims(E,1)-np.expand_dims(E,0))).sum(2)>0.001)
    sm=((M==0).sum()-M.shape[0])/2
    print('similar:',sm)