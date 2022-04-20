import torch
import torch.nn as nn
import torch.nn.functional as F

import torch_geometric.transforms as T
import numpy as np
import itertools
import torch_geometric
import scipy

def genrate_perm(perm_idx):
    perm = [np.random.permutation(perm) for perm in perm_idx]
    return perm

def generate_A(edge_index, n):
    A=np.zeros((n,n),dtype=np.float32)
    A[edge_index[0],edge_index[1]]=1
    return torch.from_numpy(A.flatten())

def sort_fn_laplacian(x,edge_index):

    # construct laplacian 
    L_e, L_w = torch_geometric.utils.get_laplacian(edge_index)
    L = np.zeros((x.shape[0],x.shape[0]),dtype=np.float32)
    L[L_e[0],L_e[1]]=L_w

    # compute eigen decomposition of Laplacian, evals are returned in ascending order
    evals, evecs = np.linalg.eigh(L)
    # ----- create sorting criterion -----  
    unique_vals, evals_idx, evals_mult = np.unique(evals, return_counts=True, return_index=True) # get eigenvals multiplicity
        
    chosen_evecs = []
    for ii in range(len(evals_idx)):
        if evals_mult[ii] == 1:
            chosen_evecs.append(np.abs(evecs[:,evals_idx[ii]]))
        else:
            eigen_space_start_idx = evals_idx[ii]
            eigen_space_size = evals_mult[ii]
            eig_space_basis = evecs[:, eigen_space_start_idx:(eigen_space_start_idx+eigen_space_size)]
            chosen_evecs.append(np.sqrt((eig_space_basis ** 2).sum(1)))

    chosen_evecs = np.stack(chosen_evecs, axis=1).round(decimals=2)
    sort_idx = np.lexsort([col for col in chosen_evecs.transpose()[::-1]]) # consider regular sort
    return sort_idx, chosen_evecs


class SortFrame(object):
    def __init__(self,device, pre_transform,sort_fn=sort_fn_laplacian):
        self.pre_transform = pre_transform
        self.sort_fn = sort_fn
        self.device = device

    def __call__(self, data):
        data = self.pre_transform(data)
        sort_idx, to_sort = self.sort_fn(data.x, data.edge_index)
        sorted_x = to_sort[sort_idx,:]
        unique_rows, dup_rows_idx, dup_rows_mult = np.unique(sorted_x, axis=0, return_index=True, return_counts=True)

        perm_start_idx = dup_rows_idx[dup_rows_mult!=1]
        perm_size = dup_rows_mult[dup_rows_mult!=1]
        perm_idx = []
        for ii in range(len(perm_size)):
            perm_idx.append(np.arange(perm_start_idx[ii], perm_start_idx[ii]+perm_size[ii]))
        data.perm_idx = perm_idx
        data.sort_idx = sort_idx
        data.size = data.x.shape[0]
        return data


class SampleFrame(object):
    def __init__(self,size=64,sample_size=10, GA=False,MLP=False):
        self.sample_size = sample_size
        self.size = size
        self.counter=0
        self.GA=GA
        self.id = not MLP
        self.MLP = MLP

    def apply_permutation(self, perm_idx, edge_index, x):
        inv_perm_idx = np.zeros_like(perm_idx)
        inv_perm_idx[perm_idx] = np.arange(perm_idx.shape[0])
        sorted_edge_index = torch.tensor(inv_perm_idx[edge_index])
        sorted_x = x[perm_idx,:]
        return sorted_edge_index, sorted_x

    def permute_with_perm(self,perm_idx, perm, sort_idx, edge_index, x):
        inv_sort_idx = np.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(sort_idx.shape[0])
        inv_sort_idx[sort_idx[list(itertools.chain(*perm_idx))]] = list(itertools.chain(*perm))
        sorted_edge_index = inv_sort_idx[edge_index]
        cur_sort_idx = np.zeros_like(sort_idx)
        cur_sort_idx[inv_sort_idx] = np.arange(sort_idx.shape[0])
     
        sorted_x_feat = x[cur_sort_idx,:]

        return sorted_edge_index, sorted_x_feat
     


    def __call__(self,data):
        n = data.x.shape[0]
        m = self.size
        x = data.x
        d = x.shape[1]
        edge_index = data.edge_index
        sort_idx = data.sort_idx
        perm_idx = data.perm_idx
        x_arr = []
        e_arr = []
        if not perm_idx:
            if self.GA:
                sorted_edge_index, sorted_x = self.apply_permutation(np.random.permutation(n), edge_index.clone(), x.clone())
            else:
                sorted_edge_index, sorted_x = self.apply_permutation(sort_idx, edge_index.clone(), x.clone())
                data.edge_index = sorted_edge_index.detach()
            if self.MLP:
                new_x = torch.cat((sorted_x.flatten(),torch.zeros((m-n,d),dtype=x.dtype),generate_A(sorted_edge_index,m)))
                data = new_x,data.y
            else:
                if self.id:
                    data.x = torch.cat([sorted_x.detach(),torch.eye(n, dtype=x.dtype),torch.zeros((n,m-n), dtype=x.dtype)],1).clone()
                else:
                    data.x = sorted_x.detach()
                data.edge_index = sorted_edge_index
        else:
            for i in range(self.sample_size):
                if self.GA:
                    sorted_edge_index, sorted_x = self.apply_permutation(np.random.permutation(n), edge_index.clone(), x.clone())
                else:
                    perm = genrate_perm(perm_idx)
                    sorted_edge_index, sorted_x = self.permute_with_perm(perm_idx, perm, sort_idx, edge_index.clone(), x.clone())
                    sorted_edge_index = torch.from_numpy(sorted_edge_index)
                if self.MLP:
                    x_arr.append(torch.cat((sorted_x.flatten(),torch.zeros((m-n,d),dtype=x.dtype).flatten(),generate_A(sorted_edge_index,m))))
                else:
                    if self.id:
                        x_arr.append(torch.cat([sorted_x.detach(),torch.eye(n, dtype=x.dtype),torch.zeros((n,m-n), dtype=x.dtype)],1).clone())
                    else:
                        x_arr.append(sorted_x.detach())
                    e_arr.append(torch.tensor(sorted_edge_index.clone()) + (i*n))
            if self.MLP:
                data = torch.stack(x_arr,dim=0), data.y
            else:
                data.x = torch.cat(x_arr,0).detach()
                data.edge_index = torch.cat(e_arr,1).detach()
        return data


class SampleFrame8C(object):
    def __init__(self,size=64,sample_size=10, GA=False,MLP=False):
        self.sample_size = sample_size
        self.size = size
        self.counter=0
        self.GA=GA
        self.id = not MLP
        self.MLP = MLP

    def apply_permutation(self, perm_idx, edge_index, x):
        inv_perm_idx = np.zeros_like(perm_idx)
        inv_perm_idx[perm_idx] = np.arange(perm_idx.shape[0])
        sorted_edge_index = torch.tensor(inv_perm_idx[edge_index])
        sorted_x = x[perm_idx,:]
        return sorted_edge_index, sorted_x

    def permute_with_perm(self,perm_idx, perm, sort_idx, edge_index, x):
        inv_sort_idx = np.zeros_like(sort_idx)
        inv_sort_idx[sort_idx] = np.arange(sort_idx.shape[0])
        inv_sort_idx[sort_idx[list(itertools.chain(*perm_idx))]] = list(itertools.chain(*perm))
        sorted_edge_index = inv_sort_idx[edge_index]
        cur_sort_idx = np.zeros_like(sort_idx)
        cur_sort_idx[inv_sort_idx] = np.arange(sort_idx.shape[0])
     
        sorted_x_feat = x[cur_sort_idx,:]

        return sorted_edge_index, sorted_x_feat
     


    def __call__(self,data):
        n = data.x.shape[0]
        m = self.size
        x = data.x
        d = x.shape[1]
        edge_index = data.edge_index
        sort_idx = data.sort_idx
        perm_idx = data.perm_idx
        x_arr = []
        e_arr = []
        if not perm_idx:
            if self.GA:
                sorted_edge_index, sorted_x = self.apply_permutation(np.random.permutation(n), edge_index.clone(), x.clone())
            else:
                sorted_edge_index, sorted_x = self.apply_permutation(sort_idx, edge_index.clone(), x.clone())
                data.edge_index = sorted_edge_index.detach()
            if self.MLP:
                new_x = generate_A(sorted_edge_index,m).unsqueeze(0)
                data = new_x,data.y
            else:
                if self.id:
                    data.x = torch.cat([sorted_x.detach(),torch.eye(n, dtype=x.dtype),torch.zeros((n,m-n), dtype=x.dtype)],1).clone()
                else:
                    data.x = sorted_x.detach()
                data.edge_index = sorted_edge_index
        else:
            for i in range(self.sample_size):
                if self.GA:
                    sorted_edge_index, sorted_x = self.apply_permutation(np.random.permutation(n), edge_index.clone(), x.clone())
                else:
                    perm = genrate_perm(perm_idx)
                    sorted_edge_index, sorted_x = self.permute_with_perm(perm_idx, perm, sort_idx, edge_index.clone(), x.clone())
                    sorted_edge_index = torch.from_numpy(sorted_edge_index)
                if self.MLP:
                    x_arr.append(generate_A(sorted_edge_index,m))
                else:
                    if self.id:
                        x_arr.append(torch.cat([sorted_x.detach(),torch.eye(n, dtype=x.dtype),torch.zeros((n,m-n), dtype=x.dtype)],1).clone())
                    else:
                        x_arr.append(sorted_x.detach())
                    e_arr.append(torch.tensor(sorted_edge_index.clone()) + (i*n))
            if self.MLP:
                data = torch.stack(x_arr,dim=0), data.y
            else:
                data.x = torch.cat(x_arr,0).detach()
                data.edge_index = torch.cat(e_arr,1).detach()
        return data
