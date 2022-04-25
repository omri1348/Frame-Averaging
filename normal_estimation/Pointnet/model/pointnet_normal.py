import logging
from numpy import matrixlib
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
from model.pointnet import STN3d, STNkd
import utils.general as utils

class PointNet(nn.Module):
    def __init__(self, num_part,with_bn,union_frame,is_leaky,is_max_pooling,is_detach_frame,is_rotation_only,frame_agg_type,is_local_frame,k_size):
        super(PointNet, self).__init__()
        
        channel = 3
        self.num_part = num_part
        self.stn = STN3d(channel,with_bn)
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 128, 1)
        self.conv4 = torch.nn.Conv1d(128, 512, 1)
        self.conv5 = torch.nn.Conv1d(512, 2048, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(128)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(2048)
        self.fstn = STNkd(k=128,with_bn=with_bn)
        self.convs1 = torch.nn.Conv1d(4944 - 16, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, num_part, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)
        self.with_bn = with_bn

    def forward(self, point_cloud, label):
        output = {}
        B, D, N = point_cloud.size()
        trans = self.stn(point_cloud)
        point_cloud = point_cloud.transpose(2, 1)
        if D > 3:
            point_cloud, feature = point_cloud.split(3, dim=2)
        point_cloud = torch.bmm(point_cloud, trans)
        if D > 3:
            point_cloud = torch.cat([point_cloud, feature], dim=2)

        point_cloud = point_cloud.transpose(2, 1)

        out1 = F.relu(self.bn1(self.conv1(point_cloud)) if self.with_bn else self.conv1(point_cloud))
        out2 = F.relu(self.bn2(self.conv2(out1)) if self.with_bn else self.conv2(out1))
        out3 = F.relu(self.bn3(self.conv3(out2)) if self.with_bn else self.conv3(out2))

        trans_feat = self.fstn(out3)
        output['trans_feat'] = trans_feat
        x = out3.transpose(2, 1)
        net_transformed = torch.bmm(x, trans_feat)
        net_transformed = net_transformed.transpose(2, 1)

        out4 = F.relu(self.bn4(self.conv4(net_transformed)))
        out5 = self.bn5(self.conv5(out4)) if self.with_bn else self.conv5(out4)
        out_max = torch.max(out5, 2, keepdim=True)[0]
        out_max = out_max.view(-1, 2048)

        expand = out_max.view(-1, 2048, 1).repeat(1, 1, N)
        concat = torch.cat([expand, out1, out2, out3, out4, out5], 1)
        net = F.relu(self.bns1(self.convs1(concat)) if self.with_bn else self.convs1(concat))
        net = F.relu(self.bns2(self.convs2(net)) if self.with_bn else self.convs2(net))
        net = F.relu(self.bns3(self.convs3(net)) if self.with_bn else self.convs3(net))
        net = self.convs4(net)
        net = net.transpose(2, 1).contiguous()
        
        net = net.view(B, N, self.num_part) # [B, N, 3]
        output['pred'] = torch.nn.functional.normalize(net,p=2,dim=-1)

        return output



