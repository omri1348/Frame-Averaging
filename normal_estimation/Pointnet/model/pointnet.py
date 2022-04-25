import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as func


class STN3d(nn.Module):
    def __init__(self, channel,with_bn):
        super(STN3d, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.with_bn = with_bn

    def forward(self, x):
        batchsize = x.size()[0]
        x = func.relu(self.bn1(self.conv1(x)) if self.with_bn else self.conv1(x))
        x = func.relu(self.bn2(self.conv2(x)) if self.with_bn else self.conv2(x))
        x = func.relu(self.bn3(self.conv3(x)) if self.with_bn else self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = func.relu(self.bn4(self.fc1(x)) if self.with_bn else self.fc1(x))
        x = func.relu(self.bn5(self.fc2(x)) if self.with_bn else self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(np.float32))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64,with_bn=True):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)
        self.with_bn = with_bn
        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = func.relu(self.bn1(self.conv1(x)) if self.with_bn else self.conv1(x))
        x = func.relu(self.bn2(self.conv2(x)) if self.with_bn else self.conv2(x))
        x = func.relu(self.bn3(self.conv3(x)) if self.with_bn else self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = func.relu(self.bn4(self.fc1(x))if self.with_bn else self.fc1(x))
        x = func.relu(self.bn5(self.fc2(x))if self.with_bn else self.fc2(x))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss