import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.nn.init as init

class Transform_Net(nn.Module):
    def __init__(self):
        super(Transform_Net, self).__init__()
        self.k = 3

        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm1d(1024)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv1d(128, 1024, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.linear1 = nn.Linear(1024, 512, bias=False)
        self.bn3 = nn.BatchNorm1d(512)
        self.linear2 = nn.Linear(512, 256, bias=False)
        self.bn4 = nn.BatchNorm1d(256)

        self.transform = nn.Linear(256, 3 * 3)
        init.constant_(self.transform.weight, 0)
        init.eye_(self.transform.bias.view(3, 3))

    def forward(self, x):
        batch_size = x.size(0)

        x = self.conv1(x)
        x = self.conv2(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = self.conv3(x)
        x = x.max(dim=-1, keepdim=False)[0]

        x = F.leaky_relu(self.bn3(self.linear1(x)), negative_slope=0.2)
        x = F.leaky_relu(self.bn4(self.linear2(x)), negative_slope=0.2)

        x = self.transform(x)
        x = x.view(batch_size, 3, 3)

        return x


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
    idx = pairwise_distance.topk(k=k, dim=-1)[1]
    return idx


def get_graph_feature(x, k=20):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)

    idx = knn(x, k=k)
    device = torch.device('cuda')
    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature


class FA_DGCNN(nn.Module):
    def __init__(self, num_part=3):
        super(FA_DGCNN, self).__init__()

        self.n_knn = 40
        self.num_part = num_part
        self.transform_net = Transform_Net()
        self.with_bn = True

        self.bn1 = nn.BatchNorm2d(64) if self.with_bn else nn.Identity()
        self.bn2 = nn.BatchNorm2d(64)  if self.with_bn else nn.Identity()
        self.bn3 = nn.BatchNorm2d(64)  if self.with_bn else nn.Identity()
        self.bn4 = nn.BatchNorm2d(64)  if self.with_bn else nn.Identity()
        self.bn5 = nn.BatchNorm2d(64)  if self.with_bn else nn.Identity()
        self.bn6 = nn.BatchNorm1d(1024)  if self.with_bn else nn.Identity()
        self.bn7 = nn.BatchNorm1d(64)  if self.with_bn else nn.Identity()
        self.bn8 = nn.BatchNorm1d(256)  if self.with_bn else nn.Identity()
        self.bn9 = nn.BatchNorm1d(256)  if self.with_bn else nn.Identity()
        self.bn10 = nn.BatchNorm1d(128)  if self.with_bn else nn.Identity()

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(64, 64, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv2d(64 * 2, 64, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv6 = nn.Sequential(nn.Conv1d(195, 1024, kernel_size=1, bias=False),
                                   self.bn6,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.conv8 = nn.Sequential(nn.Conv1d(1216, 256, kernel_size=1, bias=False),
                                   self.bn8,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp1 = nn.Dropout(p=0.5)
        self.conv9 = nn.Sequential(nn.Conv1d(256, 256, kernel_size=1, bias=False),
                                   self.bn9,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.dp2 = nn.Dropout(p=0.5)
        self.conv10 = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1, bias=False),
                                    self.bn10,
                                    nn.LeakyReLU(negative_slope=0.2))
        self.conv11 = nn.Sequential(nn.Conv1d(128, num_part, kernel_size=1, bias=False),
                                    nn.Sigmoid())

    def get_local_frame(self, pnts):
        center = pnts.mean(1, True)
        pnts_centered = pnts - center
        R = torch.bmm(pnts_centered.transpose(1, 2), pnts_centered)
        lambdas, V_ = torch.symeig(R.detach().cpu(), True)
        F = V_.to(R)
        return F.detach(), center.detach()

    def apply_frame(self, point_cloud):
        point_cloud = point_cloud.transpose(1, 2)

        self.ops = torch.tensor([[1, 1, 1],
                            [1, 1, -1],
                            [1, -1, 1],
                            [1, -1, -1],
                            [-1, 1, 1],
                            [-1, 1, -1],
                            [-1, -1, 1],
                            [-1, -1, -1]]).unsqueeze(1).to(point_cloud)

        Frame, center = self.get_local_frame(point_cloud)
        F_ops = self.ops.unsqueeze(0) * Frame.unsqueeze(1)

        framed_input = torch.einsum('boij,bpj->bopi', F_ops.transpose(2, 3), (point_cloud - center))
        s1, s2 , s3, s4 = framed_input.shape
        framed_input = framed_input.view(s1*s2, s3, s4).transpose(-1, -2)
        return framed_input, F_ops



    def forward(self, point_cloud, l):
        output = {}
        batch_size = point_cloud.size(0)
        num_points = point_cloud.size(2)

        x, F_ops = self.apply_frame(point_cloud)
        framed = x.clone()

        x0 = get_graph_feature(x, k=self.n_knn)
        t = self.transform_net(x0)
        x = x.transpose(2, 1)
        x = torch.bmm(x, t)
        x = x.transpose(2, 1)

        x = get_graph_feature(x, k=self.n_knn)
        x = self.conv1(x)
        x = self.conv2(x)
        x1 = x.mean(dim=-1, keepdim=False)

        x = get_graph_feature(x1, k=self.n_knn)
        x = self.conv3(x)
        x = self.conv4(x)
        x2 = x.mean(dim=-1, keepdim=False)

        x = get_graph_feature(x2, k=self.n_knn)
        x = self.conv5(x)
        x3 = x.mean(dim=-1, keepdim=False)

        x = torch.cat((framed, x1, x2, x3), dim=1)

        x4 = self.conv6(x)
        x4 = x4.mean(dim=-1, keepdim=True)

        x4 = x4.repeat(1, 1, num_points)


        x = torch.cat((x4, x1, x2, x3), dim = 1)

        x = self.conv8(x)
        x = self.dp1(x)
        x = self.conv9(x)
        x = self.conv10(x)
        x = self.dp2(x)
        x = self.conv11(x)*2 -1

        x = x.view(batch_size, 8, 3, -1).transpose(-1, -2)
        x = torch.einsum('boij,bopj->bopi', F_ops, x[..., :3]).mean(1)
        output['pred'] = torch.nn.functional.normalize(x, p=2, dim=-1)
        
        return output
