# Graphs - Expressive Power
This section reproduce the results of the graph separation tasks (section 5.2). 
Every .py file (graph8c/ exp_iso/ exp_classify) reproduce the results of the corresponding column from table 2 in the paper.
The hyperparameters used for reproducing the results are hardcoded and to repeat the experiment all is needed is to run the .py file.

## Usage
In order to choose the model to run (FA-GIN+ID/ GA-GIN+ID/ FA-MLP/ GA-MLP), edit line 13 and 14 in the relevant experiment's file:\\
line 13 - MODEL = 'gin' for GIN+ID and 'mlp' for MLP.\\
line 14 - FA = True for FA and FALSE for GA.

