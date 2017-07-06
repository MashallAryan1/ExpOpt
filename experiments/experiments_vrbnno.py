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

dimensions = (2, 3, 5, 10, 20, 40)# if len(argv) < 2 else eval(argv[1])

func_id = 1
dim = 4
instance = 1
struc_index =  0
cma_popsize=2
cma_max_itr=5
n_init_steps=3
n_steps= 3
# func_id =  eval(argv[0])
# dim =  eval(argv[1])
# instance = eval(argv[2])
# struc_index = eval(argv[3]) # [0,1]
# cma_popsize=20
# cma_max_itr=50
# n_init_steps=30
# n_steps= 150
alg_name = "vrbnno"+"_{}".format(str(struc_index))
alg_info = "Variational Bayesian Neural Net Optimization"
noise_std = 0.1


transfer_func = 'tanh'
# structures = [(dim,500,500,500,1), (dim,1024,1024,1024,1)] #[ (1,100,50,100,1),(1,100,10,100,1),(1,100,100,100,1),(1,100,2,100,2,100,1),(1,100,100,10,100,100,1),(1,100,100,100,100,100,1)]
structures = [(dim,5,5,5,1), (dim,10,10,10,1)] #[ (1,100,50,100,1),(1,100,10,100,1),(1,100,100,100,1),(1,100,2,100,2,100,1),(1,100,100,10,100,100,1),(1,100,100,100,100,100,1)]
structure = structures[struc_index]

exp = Experiment('bbob', func_id, dim, instance, max_num_instances=30
                  ,  n_init_steps=n_init_steps, n_steps=n_steps
                  , base_output_folder =""
                  , alg_name = alg_name
                  , alg_info = alg_info, cma_popsize=cma_popsize, cma_max_itr=cma_max_itr, model_type= Bayesian_neural_net
                             , structure=structure
                            , transfer_func=transfer_func
                            , num_training_steps=2000
                            , ensemble_size=50
                    , standardize=True, noise_std = noise_std )

print("begin")
exp.optimize()
print("end")