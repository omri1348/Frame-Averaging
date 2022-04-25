from torch import nn
import torch
import torch.nn.functional as F
# from model.pointnet import feature_transform_reguliarzer

class GenLoss(nn.Module):
    def __init__(self,):
        super().__init__()

def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1) - I), dim=(1, 2)))
    return loss

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)
        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss

class PointNetNormalLoss(GenLoss):
    def __init__(self,**kwargs):
        super().__init__()
        self.mat_diff_loss_scale = kwargs['mat_diff_loss_scale']
        

    def forward(self, output, target,epoch):
        debug = {}
        #torch.nn.functional.normalize(output['pred'],p=2,dim=-1)
        acc = (output['pred'] * target).sum(-1)
        eval = (1 - acc**2).mean(-1)
        acc = (1 - acc.abs()).mean
        loss = eval.mean()
        debug['normal_loss'] = loss

        total_loss = loss
        if 'trans_feat' in output:
            mat_diff_loss = feature_transform_reguliarzer(output['trans_feat'])
            debug['mat_diff_loss'] = mat_diff_loss
            total_loss = total_loss + mat_diff_loss * self.mat_diff_loss_scale
        debug['total_loss'] = loss
       
        return {"loss": total_loss,"eval":eval, "acc":acc,"loss_monitor":debug}


