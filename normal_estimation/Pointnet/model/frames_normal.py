import logging
from numpy import matrixlib
from pyparsing import withAttribute
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import utils.general as utils


class Custom_BatchNorm(nn.BatchNorm2d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(Custom_BatchNorm, self).forward(input.transpose(1,3)).transpose(1,3)

class Custom_BatchNorm1d(nn.BatchNorm1d):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return super(Custom_BatchNorm1d, self).forward(input.transpose(1,3).reshape(input.shape[0],input.shape[-1],-1)).view_as(input)
class Frames_Base(nn.Module):
    def __init__(self, num_part,with_bn,union_frame,is_leaky,is_max_pooling,is_detach_frame,is_rotation_only,frame_agg_type,is_local_frame,k_size):
        super(Frames_Base, self).__init__()
        self.num_part = num_part
        self.union_frame = union_frame
        self.is_leaky = is_leaky
        self.is_max_pooling = is_max_pooling
        self.is_detach_frame = is_detach_frame
        self.is_rotation_only = is_rotation_only
        self.is_local_frame = is_local_frame
        self.frame_agg_type = frame_agg_type
        self.k_size = k_size
        self.with_bn = with_bn
    
    def get_frame(self,pnts,override=None):
        if override is None:
            is_local_frame = self.is_local_frame
        else:
            is_local_frame = override
        if is_local_frame:
            batch_size = pnts.size(0)
            num_points = pnts.size(1)
            
            def knn(x, k):
               
                x_squrae = (x**2).sum(-1,True)
                pairwise_distance = -(x_squrae -2*torch.bmm(x,x.transpose(1,2))  + x_squrae.transpose(1,2))
            
                idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
                return idx
            idx = knn(pnts, k=self.k_size)
            device = torch.device('cuda')

            idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

            idx = idx + idx_base

            idx = idx.view(-1)
             
            pnts = pnts.view(batch_size*num_points, -1)[idx, :].view(batch_size, num_points, self.k_size, 3) 
                
            center = pnts.mean(2,False)
            pnts_centered = pnts - center.unsqueeze(2)
            
            R =  torch.einsum('bpki,bpkj->bpij',pnts_centered,pnts_centered)
            lambdas,V_ = torch.symeig(R.detach().cpu(),True)            
            F =  V_.to(R)
            if self.is_detach_frame:
                return F.detach(), center.detach(),pnts_centered.detach()
            else:
                return F,center,pnts_centered

        else:
            center = pnts.mean(1,True)
            pnts_centered = pnts - center
            R = torch.bmm(pnts_centered.transpose(1,2),pnts_centered)
            lambdas,V_ = torch.symeig(R.detach().cpu(),True)            
            F =  V_.to(R).unsqueeze(1).repeat(1,pnts.shape[1],1,1)
            if self.is_detach_frame:
                return F.detach(), center.detach()
            else:
                return F,center

class FA_Pointnet(Frames_Base):
    def __init__(self,num_part,with_bn,union_frame,is_leaky,is_max_pooling,is_detach_frame,is_rotation_only,frame_agg_type,is_local_frame,k_size):
        super(FA_Pointnet, self).__init__(num_part,with_bn,union_frame,is_leaky,is_max_pooling,is_detach_frame,is_rotation_only,frame_agg_type,is_local_frame,k_size)
        
        self.conv1 = torch.nn.Linear(3,64)# torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Linear(64,128)#torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Linear(128,128)#torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Linear(128,512)#torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Linear(512,2048)#torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = Custom_BatchNorm(64)
        self.bn2 = Custom_BatchNorm(128)
        self.bn3 = Custom_BatchNorm(128)
        self.bn4 = Custom_BatchNorm(512)
        self.bn5 = Custom_BatchNorm(2048)
        self.convs1 = torch.nn.Linear(4928,256)# torch.nn.Conv1d(4944 - 16, 256, 1)
        self.convs2 = torch.nn.Linear(256, 256)#torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Linear(256, 128) #torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Linear(128, self.num_part) #torch.nn.Conv1d(128, num_part, 1)
        self.bns1 = Custom_BatchNorm(256)
        self.bns2 = Custom_BatchNorm(256)
        self.bns3 = Custom_BatchNorm(128)
        

    
 

    def forward(self, point_cloud, label):
        output = {}
        B, D, N = point_cloud.size()
        point_cloud = point_cloud.transpose(1,2)
        Frame,center = self.get_frame(point_cloud,False)

        if self.is_rotation_only:
            ops = torch.tensor([[1,1],
                                [1,-1],
                                [-1,1],
                                [-1,-1]]).unsqueeze(1).to(point_cloud)
            F_ops = ops.unsqueeze(0) * Frame[:,:,:2].unsqueeze(1)
            F_ops = torch.cat([F_ops,torch.cross(F_ops[...,0],F_ops[...,1]).unsqueeze(-1)],dim=-1)
        else:
            ops = torch.tensor([[1,1,1],
                                    [1,1,-1],
                                    [1,-1,1],
                                    [1,-1,-1],
                                    [-1,1,1],
                                    [-1,1,-1],
                                    [-1,-1,1],
                                    [-1,-1,-1]]).unsqueeze(1).to(point_cloud)

            F_ops = ops.unsqueeze(0).unsqueeze(2) * Frame.unsqueeze(1)

        
        framed_input = torch.einsum('bopij,bpj->bopi',F_ops.transpose(3,4),(point_cloud - center))

        out1 = F.relu(self.bn1(self.conv1(framed_input)) if self.with_bn else self.conv1(framed_input))
        #out1 = torch.cat([torch.einsum('boij,bopj->bopi',F_ops,out1[...,:3]) + center.unsqueeze(1),out1[...,3:]],dim=-1).mean(1)

        #Frame,center = self.get_local_frame(out1[...,:3])
        #F_ops = ops.unsqueeze(0) * Frame.unsqueeze(1)
        #framed_input = torch.einsum('boij,bpj->bopi',F_ops.transpose(2,3),(out1[...,:3] - center))
        out2 = F.relu(self.bn2(self.conv2(out1)) if self.with_bn else self.conv2(out1))
        out3 = F.relu(self.bn3(self.conv3(out2)) if self.with_bn else self.conv3(out2))
        out4 = F.relu(self.bn4(self.conv4(out3)) if self.with_bn else self.conv4(out3))
        out5 = self.bn5(self.conv5(out4)) if self.with_bn else self.conv5(out4)
        
        out_max = torch.max(out5, 2, keepdim=True)[0]
        #out_max = out_max.view(-1, 2048)

        
        expand = out_max.repeat(1,1,N,1) #out_max.view(-1, 1,2048).repeat(1, N,1)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], -1)

        
        outs1 = F.relu(self.bns1(self.convs1(concat)) if self.with_bn else self.convs1(concat))
        outs2 = F.relu(self.bns2(self.convs2(outs1)) if self.with_bn else self.convs2(outs1))
    
        outs3 = F.relu(self.bns3(self.convs3(outs2)) if self.with_bn else self.convs3(outs2))

        outs4 = self.convs4(outs3)
        outs4 = (torch.einsum('bopij,bopj->bopi',F_ops,outs4)).mean(1)

        # net = F.relu(self.bns1(self.convs1(concat)))
        # net = F.relu(self.bns2(self.convs2(net)))
        # net = F.relu(self.bns3(self.convs3(net)))
        # net = self.convs4(net)
        # net = net.transpose(2, 1).contiguous()
        #net = outs4.view(-1, self.num_part)
        net = F.normalize(outs4,p=2,dim=-1)
        output['pred'] = net

        return output

class FA_LocalPointnet(Frames_Base):
    def __init__(self,num_part,with_bn,union_frame,is_leaky,is_max_pooling,is_detach_frame,is_rotation_only,frame_agg_type,is_local_frame,k_size):
        super(FA_LocalPointnet, self).__init__(num_part,with_bn,union_frame,is_leaky,is_max_pooling,is_detach_frame,is_rotation_only,frame_agg_type,is_local_frame,k_size)
        
        
        channel = 3
        
        self.conv1_e = torch.nn.Linear(channel*20, (64//3) * 3)
        self.conv2_e = torch.nn.Linear((64//3) * 3, (128//3)*3)
        self.conv3_e = torch.nn.Linear((128//3)*3, (128//3)*3)
        self.bn1_e = Custom_BatchNorm((64//3) * 3)
        self.bn2_e = Custom_BatchNorm((128//3)*3)
        self.bn3_e = Custom_BatchNorm((128//3)*3)


        self.conv1 = torch.nn.Linear((128//3)*3, 128)
        self.conv2 = torch.nn.Linear(128, 256)
        self.conv3 = torch.nn.Linear(256, 256)
        self.conv4 = torch.nn.Linear(256, 512)
        self.conv5 = torch.nn.Linear(512, 2048)
        self.bn1 = Custom_BatchNorm(128)
        self.bn2 = Custom_BatchNorm(256)
        self.bn3 = Custom_BatchNorm(256)
        self.bn4 = Custom_BatchNorm(512)
        self.bn5 = Custom_BatchNorm(2048)
        
        self.convs1 = torch.nn.Linear(5248,256)# torch.nn.Conv1d(4944 - 16, 256, 1)
        self.convs2 = torch.nn.Linear(256, 256)#torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Linear(256, 128) #torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Linear(128, self.num_part) #torch.nn.Conv1d(128, num_part, 1)
        self.bns1 = Custom_BatchNorm(256)
        self.bns2 = Custom_BatchNorm(256)
        self.bns3 = Custom_BatchNorm(128)
        
        if self.is_leaky:
            self.actvn = torch.nn.LeakyReLU(negative_slope=0.2)
        else:
            self.actvn = torch.nn.ReLU()
        
        
        

    def forward(self, point_cloud, label):
        output = {}
        B, D, N = point_cloud.size()
        point_cloud = point_cloud.transpose(1,2)
        
        Frame,center,pnts_centered = self.get_frame(point_cloud,True)
        #Frame_global,center_global = self.get_frame(point_cloud,False)

        if self.is_rotation_only:
            ops = torch.tensor([[1,1],
                                [1,-1],
                                [-1,1],
                                [-1,-1]]).unsqueeze(1).to(point_cloud)
            F_ops = ops.unsqueeze(0) * Frame[:,:,:2].unsqueeze(1)
            F_ops = torch.cat([F_ops,torch.cross(F_ops[...,0],F_ops[...,1]).unsqueeze(-1)],dim=-1)
        else:
            ops = torch.tensor([[1,1,1],
                                    [1,1,-1],
                                    [1,-1,1],
                                    [1,-1,-1],
                                    [-1,1,1],
                                    [-1,1,-1],
                                    [-1,-1,1],
                                    [-1,-1,-1]]).unsqueeze(1).to(point_cloud)

            F_ops = ops.unsqueeze(0).unsqueeze(2) * Frame.unsqueeze(1)
            #F_ops_global = ops.unsqueeze(0).unsqueeze(2) * Frame_global.unsqueeze(1)
        
        framed_input = torch.einsum('bopij,bpkj->bopki',F_ops.transpose(3,4),pnts_centered)

        out1_e = self.actvn(self.bn1_e(self.conv1_e(framed_input.reshape(framed_input.shape[0],framed_input.shape[1],framed_input.shape[2],-1))) if self.with_bn else self.conv1_e(framed_input))
        out2_e = self.actvn(self.bn2_e(self.conv2_e(out1_e)) if self.with_bn else self.conv2_e(out1_e))
        out3_e = self.bn3_e(self.conv3_e(out2_e)) if self.with_bn else self.conv3_e(out2_e)
        out3_e = out3_e.view(out3_e.shape[0],out3_e.shape[1],out3_e.shape[2],128//3,3)
        equiv_f = torch.einsum('bopij,boprj->bopri',F_ops,out3_e).mean(1)
        output['equiv_f'] = equiv_f

        F_ops_new,center_new = self.get_frame(equiv_f.view(equiv_f.shape[0],-1,3),False)
        F_ops_new = F_ops_new.view(equiv_f.shape[0:3] + (3,3))
        F_ops_new = F_ops_new.unsqueeze(1) * ops.unsqueeze(0).unsqueeze(2).unsqueeze(2)
        framed_input = torch.einsum('boprij,bprj->bopri',F_ops_new.transpose(4,5),(equiv_f - center_new.unsqueeze(2)))
        framed_input = framed_input.reshape(framed_input.shape[0],framed_input.shape[1],framed_input.shape[2],-1)
        out1 = self.actvn(self.bn1(self.conv1(framed_input)) if self.with_bn else self.conv1(framed_input))
        out2 = self.actvn(self.bn2(self.conv2(out1)) if self.with_bn else self.conv2(out1))
        out3 = self.actvn(self.bn3(self.conv3(out2)) if self.with_bn else self.conv3(out2))
        out4 = self.actvn(self.bn4(self.conv4(out3)) if self.with_bn else self.conv4(out3))
        out5 = self.bn5(self.conv5(out4)) if self.with_bn else self.conv5(out4)
        
        out_max = torch.max(out5, 2, keepdim=True)[0]
        
        expand = out_max.repeat(1,1,N,1) 
        concat = torch.cat([expand, out1, out2, out3, out4, out5], -1)

        outs1 = self.actvn(self.bns1(self.convs1(concat)) if self.with_bn else self.convs1(concat))
        outs2 = self.actvn(self.bns2(self.convs2(outs1)) if self.with_bn else self.convs2(outs1))
        outs3 = self.actvn(self.bns3(self.convs3(outs2)) if self.with_bn else self.convs3(outs2))
        outs4 = self.convs4(outs3)
        net = (torch.einsum('boprij,bopj->bopri',F_ops_new,outs4)).mean([1,3])
        output['net'] = net
        net = F.normalize(net,p=2,dim=-1)
        output['pred'] = net
        
        return output



