'''
Created on 22/06/2017

@author: mashallaryan
'''
from __future__ import absolute_import, division, print_function, unicode_literals
import sys,os,os.path # in case we want to control what to run via command line args
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from experiments.experiment import *
# from models.gaussian_process import *
# from models.bayesian_neural_net import *
from models.dropout_net import *

argv = sys.argv[1:]

#dimensions = (10, 20, 40)# if len(argv) < 2 else eval(argv[1])
func_id = 2# eval(argv[0])
dim =  4#eval(argv[1])
instance = 2#eval(argv[2])
struc_index = 0#eval(argv[3]) # [0,1]
lr_index = 0#eval(argv[4])
dr_index = 0#eval(argv[5])
cma_popsize=2
cma_max_itr=5
n_init_steps=3
n_steps= 3



# func_id =  eval(argv[0])
# dim =  eval(argv[1])
# instance = eval(argv[2])
# struc_index = eval(argv[3]) # [0,1]
# lr_index = eval(argv[4]) # [0,2]
# dr_index = eval(argv[5]) # [0,3]
# cma_popsize=20
# cma_max_itr=50
# n_init_steps=30
# n_steps= 150

dropout_rates =[ 0.05, 0.1, 0.2, 0.5]
learning_rates=[1.0e-1,1.0e-2, 1.0e-3]
# structures = [(dim,500,500,500,1), (dim,1024,1024,1024,1)] #[ (1,100,50,100,1),(1,100,10,100,1),(1,100,100,100,1),(1,100,2,100,2,100,1),(1,100,100,10,100,100,1),(1,100,100,100,100,100,1)]
structures = [(dim,5,5,5,1), (dim,10,10,10,1)] #[ (1,100,50,100,1),(1,100,10,100,1),(1,100,100,100,1),(1,100,2,100,2,100,1),(1,100,100,10,100,100,1),(1,100,100,100,100,100,1)]

learning_rate = learning_rates[lr_index]
dropout_rate  = dropout_rates[dr_index]
structure = structures[struc_index]

alg_name = "drpoutnno"+"_{}lr{}_dr{}".format(str(struc_index),str(lr_index), str(dr_index))
alg_info = "Dropout Neural Net  Optimization with  parameters :structure= {} learning_rate = {}, dropout_rate = {}".format(str(structure), str(lr_index), str(dr_index))


transfer_func = 'tanh'

exp = Experiment('bbob', func_id, dim, instance, max_num_instances=30
                ,  n_init_steps=n_init_steps, n_steps=n_steps
                , base_output_folder =""
                , alg_name = alg_name
                , alg_info = alg_info
                , cma_popsize=cma_popsize
                , cma_max_itr=cma_max_itr
                , model_type=SDropoutRegressor
                , structure=structure
                , dropout_rate=dropout_rate
                , dropout_type= "Gaussian" #{ "zero_one" or "Gaussian"}
                , use_dropout_ontest = True
                , use_batch_normalization=True
                , use_early_stopping=True
                , transfer_func=transfer_func
                , num_training_steps=30000
                , learning_rate=learning_rate
                , ensemble_size=50
                , batch_size=150
                , weight_decay=0.001
                , standardize=True )

print("begin")
exp.optimize()
print("end")