import numpy as np
import os
import torch
import sys
import logging
import time
import plotly.offline as offline
import plotly.graph_objs as go
import utils.general as utils
from training.train import BaseTrainRunner
import pandas as pd
from tqdm import tqdm
from pytorch3d.transforms import  Rotate, random_rotations

class TrainRunner(BaseTrainRunner):
    
    
    def run(self):  
        win = None
        timing_log = []
        loss_log_epoch = []
        val_loss_log_epoch = []
        lr_log_epoch = []
        logging.debug("*******************running*********")
        self.epoch = 0
        best_loss = np.inf
        for epoch in range(self.start_epoch, self.nepochs + 2):
            self.epoch = epoch

            start_epoch = time.time()
            batch_loss = 0.0
            examples_seen = 0

            if (epoch % self.conf.get_int('train.save_checkpoint_frequency') == 0 or epoch == self.start_epoch) and epoch > 0:
                self.save_checkpoints()
            if epoch % self.conf.get_int('train.test_frequency') == 0:
                
                self.network.eval()
                total_loss_seen = 0.0
                total_examples_seen = 0
                with torch.no_grad():
                    for batch_id, (points, target) in tqdm(enumerate(self.eval_dataloader), total=len(self.eval_dataloader), smoothing=0.9):
                        
                        if self.test_rot == 'o3':
                            trot = Rotate(R=random_rotations(points.shape[0]))
                            points = trot.transform_points(points)
                            target = trot.transform_points(target)
                            
                        points = points.cuda()
                        target = target.cuda()
                        points = points.transpose(2, 1)
                        output = self.network(points,None)
                        loss_res = self.loss(output,target,epoch)
                    
                        total_loss_seen += loss_res['eval'].item()
                        total_examples_seen += points.shape[0]
                val_loss_log_epoch.append(total_loss_seen / total_examples_seen)

                if total_loss_seen / total_examples_seen < best_loss:
                    best_loss = total_loss_seen / total_examples_seen
                    logging.info("best loss : {0}".format(best_loss))
                    self.save_checkpoints(True)
            self.network.train()
            if (self.adjust_lr):
                self.adjust_learning_rate(epoch)
            
            data_index = 0
            for points,normals in self.dataloader:

                if self.train_rot == 'o3':
                    trot = Rotate(R=random_rotations(points.shape[0]))
                    points = trot.transform_points(points)
                    normals = trot.transform_points(normals)
                if self.scale_augmentation:
                    points = points.data.numpy()
                    points[:,:, 0:3] = self.random_scale_point_cloud(points[:,:, 0:3])
                    points = torch.Tensor(points)
                

                points = points.cuda()
                normals = normals.cuda()               
                points = points.transpose(2, 1)
                output = self.network(points,None)    
                
                loss_res = self.loss(output,normals,epoch=epoch)
                loss = loss_res["loss"].mean()
                loss.backward()
                
                self.optimizer.step()

                if 'total_loss' in self.step_log:
                    len_step_loss = len(self.step_log['total_loss'])
                else:
                    len_step_loss = 0
                for k in loss_res['loss_monitor'].keys():
                    if k in self.step_log:
                        self.step_log[k].append(loss_res['loss_monitor'][k].mean().item())
                    else:
                        if len_step_loss > 0:
                            self.step_log[k] = [0.0]*len_step_loss + [loss_res['loss_monitor'][k].mean().item()]
                        else:
                            self.step_log[k] =[loss_res['loss_monitor'][k].mean().item()]


                batch_loss += loss_res["eval"].sum().item()
                examples_seen += points.shape[0]
                logging.info("expname : {0}".format(self.expname))
                logging.info("timestamp: {0} , epoch : {1}, data_index : {2} , loss : {3} ".format(self.timestamp,
                                                                                                                                                epoch,
                                                                                                                                                data_index,
                                                                                                                                                loss_res['loss'].mean().item()))
                
                for param in self.network.parameters():
                    param.grad = None
                
                data_index = data_index + 1
                
            lr_log_epoch.append(self.optimizer.param_groups[0]["lr"])
            loss_log_epoch.append(batch_loss / examples_seen)


            if (epoch % self.save_learning_log_freq == 0):
                trace_steploss = []
                selected_stepdata = pd.DataFrame(self.step_log)
                for x in selected_stepdata.columns:
                    if 'loss' in x:
                        trace_steploss.append(
                            go.Scatter(x=np.arange(len(selected_stepdata)), y=selected_stepdata[x], mode='lines',
                                       name=x, visible='legendonly'))

                fig = go.Figure(data=trace_steploss)

                epoch_trace = [go.Scatter(x=np.arange(epoch+1), y=loss_log_epoch, mode='lines',
                                       name='train_loss', visible='legendonly'),
                               go.Scatter(x=np.arange(epoch+1), y=val_loss_log_epoch, mode='lines',
                                       name='val_loss', visible='legendonly')]
                fig_epoch = go.Figure(data=epoch_trace)
                path = os.path.join(self.conf.get_string('train.base_path'), self.exps_folder_name, self.expname, self.timestamp)

                env = '/'.join([path, 'loss'])
                if win is None:
                    win = self.visdomer.plot_plotly(fig, env=env)
                    win_epoch = self.visdomer.plot_plotly(fig_epoch, env=env)
                else:
                    self.visdomer.plot_plotly(fig, env=env,win=win)
                    self.visdomer.plot_plotly(fig_epoch, env=env,win=win_epoch)

                self.save_learning_log(epoch_log=dict(epoch=range(self.start_epoch, epoch + 1),
                             loss_epoch=loss_log_epoch,
                             val_loss_epoch=val_loss_log_epoch,
                             lr_epoch=lr_log_epoch),
                                       step_log=self.step_log)

