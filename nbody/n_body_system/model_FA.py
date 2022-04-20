import torch
from torch import nn
from models.gcl import GCL

def create_frame(nodes, n_nodes):
    pnts = nodes[:,:3]
    v = nodes[:,3:]
    pnts = pnts.view(-1, n_nodes, 3).transpose(1,2)
    v = v.view(-1, n_nodes, 3).transpose(1,2)
    center = pnts.mean(2,True)
    pnts_centered = pnts - center
    # add noise
    R = torch.bmm(pnts_centered,pnts_centered.transpose(1,2))
    lambdas,V_ = torch.symeig(R.detach().cpu(),True)
    F =  V_.to(R)
    ops = torch.tensor([[1,1,1],
                        [1,1,-1],
                        [1,-1,1],
                        [1,-1,-1],
                        [-1,1,1],
                        [-1,1,-1],
                        [-1,-1,1],
                        [-1,-1,-1]]).unsqueeze(1).to(F)
    F_ops = ops.unsqueeze(0) * F.unsqueeze(1)
    framed_input = torch.einsum('boij,bpj->bopi',F_ops.transpose(2,3),(pnts - center).transpose(1,2))
    framed_v = torch.einsum('boij,bpj->bopi',F_ops.transpose(2,3),(v).transpose(1,2))
    framed_input = framed_input.transpose(0,1)
    framed_input = torch.reshape(framed_input,(-1,3))
    framed_v = framed_v.transpose(0,1)
    framed_v = torch.reshape(framed_v,(-1,3))
    out = torch.cat([framed_input,framed_v],dim=1)
    return out, F_ops.detach(), center.detach()

def invert_frame(pnts, F_ops, n_nodes, center):
    pnts = pnts.view(8, -1, n_nodes,3)
    pnts = pnts.transpose(0,1)
    framed_input = torch.einsum('boij,bopj->bopi',F_ops, pnts) 
    framed_input = framed_input.mean(1) 
    if center is not None:
        framed_input = framed_input + center.transpose(1,2)
    framed_input = torch.reshape(framed_input,(-1,3))
    return framed_input

def invert_latent_frame(pnts, F_ops, batch_size, n_nodes, center):
    pnts = pnts.view(8, batch_size, n_nodes, -1,3)
    pnts = pnts.transpose(0,1)
    framed_input = torch.einsum('boij,bopfj->bopfi',F_ops, pnts) 
    framed_input = framed_input.mean(1) 
    if center is not None:
        framed_input = framed_input + center.transpose(1,2).unsqueeze(-2)
    framed_input = framed_input.contiguous()
    framed_input = framed_input.view(batch_size,-1,3)
    return framed_input

def create_latent_frame(pnts,n_nodes):
    pnts = pnts.transpose(1,2)
    center = pnts.mean(2,True)

    pnts_centered = pnts - center
    
    R = torch.bmm(pnts_centered,pnts_centered.transpose(1,2))
    lambdas,V_ = torch.symeig(R.detach().cpu(),True)
    F =  V_.to(R)
    ops = torch.tensor([[1,1,1],
                        [1,1,-1],
                        [1,-1,1],
                        [1,-1,-1],
                        [-1,1,1],
                        [-1,1,-1],
                        [-1,-1,1],
                        [-1,-1,-1]]).unsqueeze(1).to(F)
    F_ops = ops.unsqueeze(0) * F.unsqueeze(1)
    framed_input = torch.einsum('boij,bpj->bopi',F_ops.transpose(2,3),(pnts - center).transpose(1,2))
    # framed_input = torch.einsum('boij,bpj->bopi',F_ops.transpose(2,3),(pnts).transpose(1,2))
    framed_input = framed_input.transpose(0,1)
    framed_input = framed_input.view(framed_input.shape[0],framed_input.shape[1],n_nodes, -1, 3)
    framed_input = torch.reshape(framed_input,(-1,framed_input.shape[-2]*3))
    return framed_input, F_ops.detach(), center.detach()


def expand_edge(edges,n_h,n_frame=8):
    row, col = edges
    addition = torch.repeat_interleave(torch.arange(n_frame)*n_h,row.shape[0]).to(row.device)
    row_expanded = row.repeat(n_frame) + addition
    col_expanded = col.repeat(n_frame) + addition
    return [row_expanded, col_expanded]


class FA_GNN(nn.Module):
    def __init__(self, input_dim, hidden_nf, device='cpu', act_fn=nn.SiLU(), n_layers=4, attention=0, recurrent=False):
        super(FA_GNN, self).__init__()
        self.hidden_nf = hidden_nf
        self.device = device
        self.n_layers = n_layers
        self.dimension_reduce = nn.ModuleList()
        for i in range(0, n_layers):
            self.add_module("gcl_%d" % i, GCL(self.hidden_nf, self.hidden_nf, self.hidden_nf, edges_in_nf=2, act_fn=act_fn, attention=attention, recurrent=recurrent))
        self.decoder = nn.Sequential(nn.Linear(hidden_nf, hidden_nf),
                              act_fn,
                              nn.Linear(hidden_nf, 3))
                            
        self.embedding = nn.Sequential(nn.Linear(input_dim, hidden_nf))
        self.to(self.device)


    def forward(self, h, edges, edge_attr=None):
        n_frame = 8
        n_nodes = 5
        batch_size = int(h.shape[0]/n_nodes)
        n_h = h.shape[0]
        edges = expand_edge(edges, n_h, n_frame)
        edge_attr = edge_attr.repeat(n_frame,1)
        h, F_ops, center = create_frame(h, n_nodes)
        h = self.embedding(h)
        for i in range(self.n_layers):
            h, _ = self._modules["gcl_%d" % i](h, edges, edge_attr=edge_attr)
            if i < (self.n_layers - 1):
                # transform equiv features extraction
                h = invert_latent_frame(h, F_ops, batch_size, n_nodes, None)
                # compute new frame
                h, F_ops, _ = create_latent_frame(h, n_nodes)
               

        h = self.decoder(h)
        h = invert_frame(h, F_ops, n_nodes, center)
        return h



