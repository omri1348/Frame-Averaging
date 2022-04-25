import torch
import torch.utils.data as data
import numpy as np
import os
import sys
import time
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple
sys.path.append('../code')
from utils.general import *
# from utils.plots import plot_threed_scatter
import utils.general as utils
import logging
# import utils.plots as  plt
from trimesh.sample import sample_surface
import utils.general as utils
import json
import logging

class ABCDataSet(data.Dataset):

    def __init__(self,**kwargs):
        base_dir = kwargs['dataset_path']
        self.preload = kwargs['preload']
        test_split_file = './confs/splits/{0}'.format(kwargs['split'])
        with open(test_split_file, "r") as f:
            split_file = json.load(f)
        self.files = self.get_instance_filenames(base_dir,kwargs['preload'],split_file)
        self.num_of_points = kwargs['num_of_points']
        self.normalize_std = kwargs['normalize_std']
        self.is_seed_data=kwargs["is_seed_data"]
        

    def get_instance_filenames(self,base_dir,preload,split,ext='',format='obj'):
        files = []
        l = 0
        for dataset in split:
            for class_name in split[dataset]:
                for instance_name in split[dataset][class_name]:
                    
                    instance_filename = os.path.join(base_dir, "{0}{1}.{2}".format(instance_name,ext,format))
                    if not os.path.isfile(instance_filename):
                        logging.error('Requested non-existent file "' + instance_filename + " {0} ".format(l))
                    l = l+1
                    if preload:
                        mesh = trimesh.load(instance_filename)
                    else:
                        mesh = None
                    files.append((instance_filename,mesh))
        return files

    def __getitem__(self, index):
        if self.is_seed_data:
            np.random.seed(1)
        if self.preload:
            mesh = self.files[index][-1]
        else:
            mesh = trimesh.load(self.files[index][0])

        sample = sample_surface(mesh,self.num_of_points)
        points =  torch.tensor(sample[0]).float()
        points = points - points.mean(0,True)
        points = points / torch.sqrt(((torch.norm(points,p=2,dim=-1) - torch.norm(points,p=2,dim=-1).mean(0,True))**2).mean()) 
        
        normals = torch.tensor(mesh.face_normals[sample[1]]).float()
        
        return points,normals

    def __len__(self):
        return len(self.files)
