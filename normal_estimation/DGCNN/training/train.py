import utils.general as utils
import os
from datetime import datetime
from pyhocon import ConfigFactory
import sys
import torch
import numpy as np
import json
import logging
import pandas as pd
import GPUtil
from utils.visdomer import Visdomer
from itertools import chain
import socket

class MultiEpochsDataLoader(torch.utils.data.DataLoader):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._DataLoader__initialized = False
        self.batch_sampler = _RepeatSampler(self.batch_sampler)
        self._DataLoader__initialized = True
        self.iterator = super().__iter__()

    def __len__(self):
        return len(self.batch_sampler.sampler)

    def __iter__(self):
        for i in range(len(self)):
            yield next(self.iterator)


class _RepeatSampler(object):
    """ Sampler that repeats forever.
    Args:
        sampler (Sampler)
    """

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            yield from iter(self.sampler)

class BaseTrainRunner():
    def __init__(self,**kwargs):

        if (type(kwargs['conf']) == str):
            self.conf = ConfigFactory.parse_file(kwargs['conf'])
            self.conf_filename = kwargs['conf']
        else:
            self.conf = kwargs['conf']
        self.batch_size = kwargs['batch_size']
        self.nepochs = kwargs['nepochs']
        self.expnameraw = self.conf.get_string('train.expname')
        self.expname = self.conf.get_string('train.expname') +  kwargs['expname']
        debug_conf = {}
        debug_conf['batch_size'] = str(self.batch_size)

        self.step_log = {}
        self.epoch_log = {}

        base_path = self.conf.get_string('train.base_path')


        if kwargs['is_continue'] and kwargs['timestamp'] == 'latest':
            if os.path.exists(os.path.join(base_path,kwargs['exps_folder_name'],self.expname)):
                timestamps = os.listdir(os.path.join(base_path,kwargs['exps_folder_name'],self.expname))
                if (len(timestamps)) == 0:
                    is_continue = False
                    timestamp = None
                else:
                    timestamp = None
                    is_continue = True
            else:
                is_continue = False
                timestamp = None
        else:
            timestamp = kwargs['timestamp']
            is_continue = kwargs['is_continue']


        self.adjust_lr = self.conf.get_bool('train.adjust_lr')

        if kwargs['gpu_index'] == "auto":
            deviceIDs = GPUtil.getAvailable(
                order="memory",
                limit=1,
                maxLoad=0.5,
                maxMemory=0.5,
                includeNan=False,
                excludeID=[],
                excludeUUID=[],
            )
            gpu = deviceIDs[0]
        else:
            gpu = kwargs['gpu_index']
        self.GPU_INDEX = gpu
        self.exps_folder_name = kwargs['exps_folder_name']

        utils.mkdir_ifnotexists(os.path.join(base_path,self.exps_folder_name))

        self.expdir = os.path.join(base_path, self.exps_folder_name, self.expname)
        utils.mkdir_ifnotexists(self.expdir)
        self.timestamp = '{:%Y_%m_%d_%H_%M_%S}'.format(datetime.now())
        utils.mkdir_ifnotexists(os.path.join(self.expdir, self.timestamp))
        log_dir = os.path.join(self.expdir, self.timestamp, 'log')
        self.log_dir = log_dir
        utils.mkdir_ifnotexists(log_dir)
        utils.configure_logging(kwargs['debug'],kwargs['quiet'],os.path.join(self.log_dir,'log.txt'))

        
        self.visdom_env = '~'.join([self.exps_folder_name.replace('_','-'), self.expname.replace('_','-'), self.timestamp.replace('_','-')])
        

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.checkpoints_path = os.path.join(self.expdir, self.timestamp, 'checkpoints')
        utils.mkdir_ifnotexists(self.checkpoints_path)

        self.model_params_subdir = "ModelParameters"
        self.optimizer_params_subdir = "OptimizerParameters"

        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path,self.model_params_subdir))
        utils.mkdir_ifnotexists(os.path.join(self.checkpoints_path, self.optimizer_params_subdir))

        if (not self.GPU_INDEX == 'ignore'):
            os.environ["CUDA_VISIBLE_DEVICES"] = '{0}'.format(self.GPU_INDEX)
            logging.info("Set gpu : {0}".format(self.GPU_INDEX))
        else:
            logging.info("No gpu selected")

        logging.info("torch available gpus : {0}".format(torch.cuda.device_count()))

        # Backup code
        self.code_path = os.path.join(self.expdir, self.timestamp, 'code')
        utils.mkdir_ifnotexists(self.code_path)
        for folder in ['training','preprocess','evaluate','utils','model','datasets','confs','params_search','exp_prep']:
            utils.mkdir_ifnotexists(os.path.join(self.code_path, folder))
            os.system("""cp -r ./{0}/* "{1}" """.format(folder,os.path.join(self.code_path, folder)))

        os.system("""cp -r {0} "{1}" """.format(kwargs['conf'], os.path.join(self.code_path, 'confs/runconf.conf')))

        logging.info('shell command : {0}'.format(' '.join(sys.argv)))

        self.save_learning_log_freq = self.conf.get_int('train.save_learning_log_freq')
        self.learning_log_epoch_path = os.path.join(self.log_dir, 'learning_epoch_log.csv')
        self.learning_log_step_path = os.path.join(self.log_dir, 'learning_step_log.csv')
        self.debug_log_conf_path = os.path.join(self.log_dir, 'debug_conf.csv')
        self.scale_augmentation=self.conf.get_bool('train.scale_augmentation')
        
      
        self.debug_pred = self.conf.get_bool('train.debug_pred')
        self.is_seed_data = self.conf.get_bool('train.is_seed_data')
        self.ds = utils.get_class(self.conf.get_string('train.dataset_train.class'))(is_seed_data=self.is_seed_data,**self.conf.get_config('train.dataset_train.properties'))

        self.eval_ds = utils.get_class(self.conf.get_string('train.dataset_test.class'))(is_seed_data=self.is_seed_data,**self.conf.get_config('train.dataset_test.properties'))
                                                                            
        logging.info('after creating data set')
        self.dataloader = MultiEpochsDataLoader(self.ds,
                                                      batch_size=self.batch_size,
                                                      shuffle=True,
                                                      num_workers=kwargs['workers'],drop_last=True,pin_memory=True)
        self.eval_dataloader = torch.utils.data.DataLoader(self.eval_ds,
                                                      batch_size=1,
                                                      shuffle=False,
                                                      num_workers=0, drop_last=False)

        self.ds_len = len(self.ds)
        logging.info("data set size : {0}".format(self.ds_len))
        
        logging.info ("loading network class : {0}".format(self.conf.get_string('train.network_class')))
        self.network = utils.get_class(self.conf.get_string('train.network_class'))(**self.conf.get_config('network.properties'))
        self.loss = utils.get_class(self.conf.get_string('network.loss.loss_type'))(**self.conf.get_config('network.loss.properties'))
        if kwargs['parallel']:
            self.network = torch.nn.DataParallel(self.network)
            self.loss = torch.nn.DataParallel(self.loss)
            logging.info("GPU parallel mode")
        else:
            logging.info("no parallel")
        self.network = utils.get_cuda_ifavailable(self.network)
        self.loss = utils.get_cuda_ifavailable(self.loss)
        self.parallel = kwargs['parallel']
        self.lr_schedules = BaseTrainRunner.get_learning_rate_schedules(self.conf.get_list('train.learning_rate_schedule'))

        opt_params = [{"params": self.network.parameters(), "lr": self.lr_schedules[0].get_learning_rate(0)}]
        self.optimizer = torch.optim.Adam(opt_params)

        self.plots_dir = os.path.join(self.expdir, self.timestamp, 'plots')
        utils.mkdir_ifnotexists(self.plots_dir)
        self.visdomer = Visdomer(self.conf.get_string('train.visdom_server'), expname=self.expname, timestamp=self.timestamp,port=self.conf.get_int('train.visdom_port'), do_vis=kwargs['vis'])
        self.window = [None,None]
        is_loaded = False
        self.train_rot = self.conf.get_string('train.train_rot')
        self.test_rot = self.conf.get_string('train.test_rot')
        self.start_epoch = 0
        if is_continue:

            if kwargs['timestamp'] == 'latest':
                potential_timestamps = ['{:%Y_%m_%d_%H_%M_%S}'.format(t) for t in sorted([datetime.strptime(t, '%Y_%m_%d_%H_%M_%S') for t in  os.listdir(os.path.join(base_path,kwargs['exps_folder_name'],self.expname)) if not 'DS_Store' in t],reverse=True)]
            else:
                potential_timestamps = [timestamp]

            i = 0
            while not is_loaded and i < len(potential_timestamps):
                timestamp = potential_timestamps[i]
                old_checkpnts_dir = os.path.join(self.expdir, timestamp, 'checkpoints')

                if os.path.isfile(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth")):
                    try:

                        
                        saved_model_state = torch.load(os.path.join(old_checkpnts_dir, 'ModelParameters', str(kwargs['checkpoint']) + ".pth"),map_location=torch.device('cpu'))
                        self.network.load_state_dict({k:v for k,v in saved_model_state['model_state_dict'].items() if not 'temp' in k})
                        logging.info('after loading model')

                        data = torch.load(os.path.join(old_checkpnts_dir, 'OptimizerParameters', str(kwargs['checkpoint']) + ".pth"))
                        self.optimizer.load_state_dict(data["optimizer_state_dict"])
                        logging.info('after loading optimizer')
                        self.start_epoch = saved_model_state['epoch']
                        is_loaded = True

                        src = os.path.join(old_checkpnts_dir, 'OptimizerParameters', "best.pth")
                        to = os.path.join(self.checkpoints_path, self.optimizer_params_subdir)
                        os.system("""cp  {0} "{1}" """.format(src, to))

                        src = os.path.join(old_checkpnts_dir, 'ModelParameters', "best.pth")
                        to = os.path.join(self.checkpoints_path, self.model_params_subdir)
                        os.system("""cp  {0} "{1}" """.format(src, to))
                        

                        #to = os.path.join(self.checkpoints_path, self.model_params_subdir, "best.pth"))
            
                        if os.path.isfile(os.path.join(self.expdir, timestamp, 'log', 'learning_epoch_log.csv')):
                            self.epoch_log = pd.read_csv(
                                os.path.join(self.expdir, timestamp, 'log', 'learning_epoch_log.csv')).to_dict()
                            self.epoch_log.pop('Unnamed: 0', None)
                            for k in self.epoch_log.keys():
                                self.epoch_log[k] = list(self.epoch_log[k].values())

                            a = pd.DataFrame(self.epoch_log)
                            a = a[a.epoch <= self.start_epoch]
                            self.epoch_log = a.to_dict()
                            pd.DataFrame(self.epoch_log).to_csv(self.learning_log_epoch_path)
                            
                        else:
                            self.epoch_log = {}

                        if os.path.isfile(os.path.join(self.expdir, timestamp, 'log', 'learning_step_log.csv')):
                            self.step_log = pd.read_csv(
                                os.path.join(self.expdir, timestamp, 'log', 'learning_step_log.csv')).to_dict()
                            for k in self.step_log.keys():
                                self.step_log[k] = list(self.step_log[k].values())

                            self.step_log.pop('Unnamed: 0', None)
                            pd.DataFrame(self.step_log).to_csv(self.learning_log_step_path)
                        else:
                            self.step_log = {}
                    except Exception as e:
                        logging.info ('something went wrong in load timestamp : {0}'.format(timestamp))
                        logging.info (str(e))
                        i = i + 1
                else:
                    i = i + 1

        if not is_loaded:
            logging.info("---------NO TIMESTAMP LOADED------------------")

        logging.info('hostname : {0}'.format(socket.gethostname()))


    def get_learning_rate_schedules(schedule_specs):

        schedules = []

        for schedule_specs in schedule_specs:

            if schedule_specs["Type"] == "Step":
                schedules.append(
                    StepLearningRateSchedule(
                        schedule_specs["Initial"],
                        schedule_specs["Interval"],
                        schedule_specs["Factor"],
                    )
                )

            else:
                raise Exception(
                    'no known learning rate schedule of type "{}"'.format(
                        schedule_specs["Type"]
                    )
                )

        return schedules

    def adjust_learning_rate(self, epoch):
        for i, param_group in enumerate(self.optimizer.param_groups):
            param_group["lr"] = self.lr_schedules[i].get_learning_rate(epoch)

    def save_checkpoints(self,is_best=False):

        

        if is_best:
            torch.save(
            {"epoch": self.epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, "best.pth"))
            torch.save(
            {"epoch": self.epoch, "optimizer_state_dict": self.optimizer.state_dict()},
            os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "best.pth"))
        else:
            torch.save(
            {"epoch": self.epoch, "model_state_dict": self.network.state_dict()},
            os.path.join(self.checkpoints_path, self.model_params_subdir, str(self.epoch) + ".pth"))
            torch.save(
                {"epoch": self.epoch, "model_state_dict": self.network.state_dict()},
                os.path.join(self.checkpoints_path, self.model_params_subdir, "latest.pth"))

            torch.save(
                {"epoch": self.epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, str(self.epoch) + ".pth"))
            torch.save(
                {"epoch": self.epoch, "optimizer_state_dict": self.optimizer.state_dict()},
                os.path.join(self.checkpoints_path, self.optimizer_params_subdir, "latest.pth"))


    def save_learning_log(self, epoch_log,step_log):
        pd.DataFrame(epoch_log).to_csv(self.learning_log_epoch_path)
        pd.DataFrame(step_log).to_csv(self.learning_log_step_path)


    def shift_point_cloud(self,batch_data, shift_range=0.1):
        """ Randomly shift point cloud. Shift is per point cloud.
            Input:
            BxNx3 array, original batch of point clouds
            Return:
            BxNx3 array, shifted batch of point clouds
        """
        B, N, C = batch_data.shape
        shifts = np.random.uniform(-shift_range, shift_range, (B,3))
        for batch_index in range(B):
            batch_data[batch_index,:,:] += shifts[batch_index,:]
        return batch_data


    def random_scale_point_cloud(self,batch_data, scale_low=0.8, scale_high=1.25):
        """ Randomly scale the point cloud. Scale is per point cloud.
            Input:
                BxNx3 array, original batch of point clouds
            Return:
                BxNx3 array, scaled batch of point clouds
        """
        B, N, C = batch_data.shape
        scales = np.random.uniform(scale_low, scale_high, B)
        for batch_index in range(B):
            batch_data[batch_index,:,:] *= scales[batch_index]
        return batch_data



class LearningRateSchedule:
    def get_learning_rate(self, epoch):
        pass


class StepLearningRateSchedule(LearningRateSchedule):
    def __init__(self, initial, interval, factor):
        self.initial = initial
        self.interval = interval
        self.factor = factor

    def get_learning_rate(self, epoch):
        return np.maximum(self.initial * (self.factor ** (epoch // self.interval)),1.0e-5)

