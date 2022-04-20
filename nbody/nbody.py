import argparse
import torch
from n_body_system.dataset_nbody import NBodyDataset
from n_body_system.model_FA import  FA_GNN
import os
from torch import nn, optim
import json
import time
import GPUtil
import numpy as np


def set_gpu(gpu):
    if gpu[0] == '-':
        deviceIDs = []
        count = 0
        num_gpus = int(gpu[1:])
        print('searching for available gpu...')
        deviceIDs = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                            excludeUUID=[])
        print(deviceIDs)
        while len(deviceIDs) < num_gpus:
            time.sleep(60)
            count += 60
            print('Pending... | {} mins |'.format(count//60))
            deviceIDs = GPUtil.getAvailable(order='memory', limit=8, maxLoad=0.5, maxMemory=0.5, includeNan=False, excludeID=[],
                                            excludeUUID=[])
            print(deviceIDs)
        gpu = deviceIDs[0:num_gpus]
    os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(gpu)[1:-1]#.replace(' ','')

parser = argparse.ArgumentParser(description='VAE MNIST Example')
parser.add_argument('--exp_name', type=str, default='exp_1', metavar='N', help='experiment_name')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=1, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test_interval', type=int, default=5, metavar='N',
                    help='how many epochs to wait before logging test')
parser.add_argument('--outf', type=str, default='n_body_system/logs', metavar='N',
                    help='folder to output vae')
parser.add_argument('--lr', type=float, default=1e-3, metavar='N',
                    help='learning rate')
parser.add_argument('--nf', type=int, default=60, metavar='N',
                    help='learning rate')
parser.add_argument('--n_layers', type=int, default=4, metavar='N',
                    help='number of layers for the autoencoder')
parser.add_argument('--max_training_samples', type=int, default=3000, metavar='N',
                    help='maximum amount of training samples')
parser.add_argument('--dataset', type=str, default="nbody_small", metavar='N',
                    help='nbody_small, nbody')
parser.add_argument('--time_exp', type=int, default=0, metavar='N',
                    help='timing experiment')
parser.add_argument('--weight_decay', type=float, default=1e-12, metavar='N',
                    help='timing experiment')



torch.set_num_threads(1)
time_exp_dic = {'time': 0, 'counter': 0}

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
set_gpu('-1')
device = torch.device("cuda:"+os.environ["CUDA_VISIBLE_DEVICES"] if args.cuda else "cpu")
loss_mse = nn.MSELoss()

try:
    os.makedirs(args.outf)
except OSError:
    pass

try:
    os.makedirs(args.outf + "/" + args.exp_name)
except OSError:
    pass


def main():
    dataset_train = NBodyDataset(partition='train', dataset_name=args.dataset,
                                 max_samples=args.max_training_samples)
    loader_train = torch.utils.data.DataLoader(dataset_train, batch_size=args.batch_size, shuffle=True, drop_last=True)

    dataset_val = NBodyDataset(partition='val', dataset_name="nbody_small")
    loader_val = torch.utils.data.DataLoader(dataset_val, batch_size=args.batch_size, shuffle=False, drop_last=False)

    dataset_test = NBodyDataset(partition='test', dataset_name="nbody_small")
    loader_test = torch.utils.data.DataLoader(dataset_test, batch_size=args.batch_size, shuffle=False, drop_last=False)

    model = FA_GNN(input_dim=6, hidden_nf=args.nf, n_layers=args.n_layers, device=device, recurrent=True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    results =  {'epochs': [], 'train_losess': [], 'val_losess': [], 'test_losess': [], 'best_val': 1e10, 'best_test': 1e10, 'best_epoch': 0}
    best_val_loss = 1e8
    best_test_loss = 1e8
    best_epoch = 0
    for epoch in range(0, args.epochs):
        train_loss = train(model, optimizer, epoch, loader_train)
        if epoch % args.test_interval == 0:
            val_loss = train(model, optimizer, epoch, loader_val, backprop=False)
            test_loss = train(model, optimizer, epoch, loader_test, backprop=False)
            results['epochs'].append(epoch)
            results['train_losess'].append(train_loss)
            results['val_losess'].append(val_loss)
            results['test_losess'].append(test_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_test_loss = test_loss
                best_epoch = epoch
                results['best_val'] = val_loss
                results['best_test'] = test_loss
                results['best_epoch'] = epoch
            print("*** Best Val Loss: %.5f \t Best Test Loss: %.5f \t Best epoch %d" % (best_val_loss, best_test_loss, best_epoch))
        json_object = json.dumps(results, indent=4)
        with open(args.outf + "/" + args.exp_name + "/losess.json", "w") as outfile:
            outfile.write(json_object)
    return best_val_loss, best_test_loss, best_epoch

def train(model, optimizer, epoch, loader, backprop=True):
    if backprop:
        model.train()
    else:
        model.eval()

    res = {'epoch': epoch, 'loss': 0, 'coord_reg': 0, 'counter': 0}

    for batch_idx, data in enumerate(loader):
        batch_size, n_nodes, _ = data[0].size()
        data = [d.to(device) for d in data]
        data = [d.view(-1, d.size(2)) for d in data]
        loc, vel, edge_attr, charges, loc_end = data

        edges = loader.dataset.get_edges(batch_size, n_nodes)
        edges = [edges[0].to(device), edges[1].to(device)]

        optimizer.zero_grad()

        rows, cols = edges
        loc_dist = torch.sum((loc[rows] - loc[cols])**2, 1).unsqueeze(1)  # relative distances among locations
        edge_attr = torch.cat([edge_attr, loc_dist], 1).detach()  # concatenate all edge properties.
        nodes = torch.cat([loc.detach(), vel], dim=1)
        loc_pred = model(nodes, edges, edge_attr)

        loss = loss_mse(loc_pred, loc_end)
        if backprop:
            loss.backward()
            optimizer.step()

        res['loss'] += loss.item()*batch_size
        res['counter'] += batch_size

    if not backprop:
        prefix = "==> "
    else:
        prefix = ""
    print('%s epoch %d avg loss: %.5f' % (prefix+loader.dataset.partition, epoch, res['loss'] / res['counter']))

    return res['loss'] / res['counter']





if __name__ == "__main__":
    main()




