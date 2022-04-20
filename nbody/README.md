# Graphs - n-Body Problem
This section reproduces the results of the FA-GNN model in the n-body experiment (section 5.3).  
The dataset files (.npy files) can be found in n_body_system/dataset.

## Usage

To reproduce the results from the paper run the following line:  
``` python
python nbody.py --exp_name exp --nf 60--lr 1e-3  
``` 

A log file that tracks the training/validation/test losses will be opened at n_body_system/logs/exp/losses.json