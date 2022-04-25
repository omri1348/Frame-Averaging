# ABC Normal Estimation
This section reproduces the results of the FA model in the normal estimation experiment (section 5.1).  

## data
to download and extract the data (only 10k dataset) please run the following script:
``` 
bash data.sh  
``` 

## Usage
The implementation includes two versiones of backbones architectures: Pointnet and DGCNN.
In order to run a specific backbone, change to the relevant folder:
``` 
cd Pointnet
``` 
or
``` 
cd DGCNN
``` 

Different models are defined by the configuration files found in the ``confs/`` folder. 
``abc_baseleine.conf`` - Baseline model (no frame averaging)  
``abc_FA.conf`` - Frame averaging + baseline and aligned train set / aligned test set.  
``abc_FA_i_so3.conf`` - Frame averaging + baseline and aligned train set / randomly rotated test set.  
``abc_FA.conf_so3_so3.conf`` - Frame averaging + baseline and randomly rotated train set / randomly rotated test set.  
For Pointnet models there is also an option to use the local frame model, using the'FA_LOCAL' conf files.  

To use a specific model, for example a Pointnet backbone with FA and i/so3 setting, run the following line:
```python 
cd Pointnet
python training/exp_runner.py --conf ./confs/abc_FA_i_so3.conf
```