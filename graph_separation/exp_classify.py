
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential, Linear
from torch_geometric.data import DataLoader
from torch_geometric.nn import (GINConv,global_mean_pool)

import numpy as np
from libs.utils import PlanarSATPairsDataset,SpectralDesign
from frame_transform import SortFrame, SampleFrame

MODEL = 'gin' # change to 'mlp' to eval MLP model
FA = True # set false to eva; GA model
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


transform = SpectralDesign(nmax=64,recfield=1,dv=2,nfreq=5,adddegree=True)
dataset = PlanarSATPairsDataset(root="dataset/EXP/",transform=SampleFrame(size=64, sample_size=1, GA=not FA , MLP=MODEL=='mlp'),pre_transform=SortFrame(device, transform))

val_loader   = DataLoader(dataset[0:200], batch_size=100, shuffle=False)
test_loader  = DataLoader(dataset[200:400], batch_size=100, shuffle=False)
train_loader = DataLoader(dataset[400:1200], batch_size=100, shuffle=True)



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
        self.fc2 = torch.nn.Linear(10, 1)
        

    def forward(self, data):

        x=data.x 
        edge_index=data.edge_index
        
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))        
        x = F.relu(self.conv3(x, edge_index))             

        x = global_mean_pool(x, data.batch)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MlpNet(nn.Module):
    def __init__(self):
        super(MlpNet, self).__init__()
        input_dim = 4224
        self.conv1 = torch.nn.Linear(input_dim, 2048)
        self.conv2 = torch.nn.Linear(2048, 4096)
        self.conv3 = torch.nn.Linear(4096, 2048) 
        self.fc1 = torch.nn.Linear(2048, 10)
        self.fc2 = torch.nn.Linear(10, 1)
        
    def forward(self, data):

        x=data
        x = F.relu(self.conv1(x))                
        x = F.relu(self.conv2(x))        
        x = F.relu(self.conv3(x)) 
        x = x.mean(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)
        



# select your model
if MODEL == 'mlp':
    model = MlpNet().to(device) 
else:
    model = GinNet().to(device)


# be sure PPGN's bias are initialized by zero
def weights_init(m):
    if isinstance(m, nn.Conv2d):
        torch.nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            torch.nn.init.zeros_(m.bias)         
                
model.apply(weights_init)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(epoch):
    model.train()
    
    L=0
    correct=0
    for data in train_loader:
        optimizer.zero_grad()
        if MODEL == 'mlp':
            data, y_grd = data
            data = data.to(device)
            y_grd= (y_grd).type(torch.float).to(device).squeeze(-1)
        else:
            data = data.to(device)
            y_grd= (data.y).type(torch.float) 
        pre=model(data)
        pred=torch.sigmoid(pre)              
        lss=F.binary_cross_entropy(pred, y_grd.unsqueeze(-1),reduction='sum')
        
        lss.backward()
        optimizer.step()
        
        correct += torch.round(pred[:,0]).eq(y_grd).sum().item()

        L+=lss.item()
    return correct/800,L/800

def test():
    model.eval()
    correct = 0
    L=0
    for data in test_loader:
        if MODEL == 'mlp':
            data, y_grd = data
            data = data.to(device)
            y_grd= (y_grd).type(torch.float).to(device).squeeze(-1)
        else:
            data = data.to(device)
            y_grd= (data.y).type(torch.float) 
        pre=model(data)
        pred=torch.sigmoid(pre)
        correct += torch.round(pred[:,0]).eq(y_grd).sum().item()

        
        lss=F.binary_cross_entropy(pred, y_grd.unsqueeze(-1),reduction='sum')
        L+=lss.item()
    L=L/200
    s1= correct / 200
    correct = 0
    Lv=0
    for data in val_loader:
        if MODEL == 'mlp':
            data, y_grd = data
            data = data.to(device)
            y_grd= (y_grd).type(torch.float).to(device).squeeze(-1)
        else:
            data = data.to(device)
            y_grd= (data.y).type(torch.float) 
        pre=model(data)
        pred=torch.sigmoid(pre)
        correct += torch.round(pred[:,0]).eq(y_grd).sum().item()

        lss=F.binary_cross_entropy(pred, y_grd.unsqueeze(-1),reduction='sum')
        Lv+=lss.item()
    s2= correct / 200    
    Lv=Lv/200

    return s1,L, s2, Lv

bval=1000
btest=0
for epoch in range(1, 200):
    tracc,trloss=train(epoch)
    test_acc,test_loss,val_acc,val_loss = test()
    if bval>val_loss:
        bval=val_loss
        btest=test_acc    
    print('Epoch: {:02d}, trloss: {:.4f}, tracc: {:.4f}, Valloss: {:.4f}, Val acc: {:.4f},Testloss: {:.4f}, Test acc: {:.4f},best test acc: {:.4f}'.format(epoch,trloss,tracc,val_loss,val_acc,test_loss,test_acc,btest))




