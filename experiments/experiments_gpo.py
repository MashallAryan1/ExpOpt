'''
Created on 22/06/2017

@author: mashallaryan
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import sys,os,os.path # in case we want to control what to run via command line args
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from experiments.experiment import *
from models.gaussian_process import *
from models.bayesian_neural_net import *
from models.dropout_net import *

argv = sys.argv[1:]

#dimensions = (2, 3, 5, 10, 20, 40)# if len(argv) < 2 else eval(argv[1])
func_id = 1 #eval(argv[0]) # [1,24]
dim =  4#eval(argv[1]) # any integer >=1
instance = 1#eval(argv[2]) # for example between [1,30]
cma_popsize = 10
cma_max_itr = 10
n_init_steps = 10
n_steps = 10
# func_id =  eval(argv[0]) # [1,24]
# dim =  eval(argv[1]) # any integer >=1
# instance = eval(argv[2]) # for example between [1,30]
# cma_popsize = 20
# cma_max_itr = 50
# n_init_steps = 30
# n_steps = 150
alg_name = "gpo"
alg_info = "vanila GPO/EGO"
noise_std = 0.1
exp = Experiment('bbob', func_id, dim, instance, max_num_instances=30
                  ,  n_init_steps=n_init_steps
                  , n_steps=n_steps
                  , base_output_folder =""
                  , alg_name = alg_name
                  , alg_info = alg_info, cma_popsize=cma_popsize, cma_max_itr=cma_max_itr, model_type= GPRegressor
                  , num_training_steps=100
                  , standardize=True, noise_std = noise_std )


print("begin")
exp.optimize()
print("end")