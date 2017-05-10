from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
from bboptim.optimization import *
from models.bayesian_neural_net import Bayesian_neural_net
import unittest
import numpy as np
from mashall.functions import *
from matplotlib import pyplot as plt
np.random.seed(102)
def g(x):
    return np.sum(x*np.sin(x))

# bounding box
box = np.array([[-3],[3]]) 
# number of training points
n_training = 5
# dimensionality
dim = 1
#bounding box limits for all variables
low = -20
high = 20
# number of test points
n_test = 500

# boundingbox
box = [np.array(dim*[low]), np.array(dim*[high])]
#we need a set of training points. in this case 2 sets
xt = np.random.uniform(box[0], box[1], (n_training, len(box[0])))

evaluation_function = g#cma.fcts.rosen
# get the obzervation for the training points
yt = np.array([[evaluation_function(i)] for i in xt])

x = np.linspace(low, high, n_test*dim).reshape(n_test,dim)
y=  np.array([[evaluation_function(i)] for i in x])



#parameters configurations:
noise_std = 0.01


transfer_funcs = ['relu','tanh']
structures =[(1,10,10,1)] #[ (1,100,50,100,1),(1,100,10,100,1),(1,100,100,100,1),(1,100,2,100,2,100,1),(1,100,100,10,100,100,1),(1,100,100,100,100,100,1)]

for transfer_func in transfer_funcs:
    for i,structure in enumerate(structures):
        # for the description of parameters goto "feed_forward_net.py" and optimization.py
        bbopt = Optimizer( bounding_box=box, init_observatins=[xt,yt]
                            , evaluation_function = g
                            , cma_popsize=20, cma_max_itr=1, model_type=Bayesian_neural_net
                            , structure=structure 
                            , transfer_func=transfer_func
                            , num_training_steps=2000
                            , ensemble_size=30
                            , standardize=True)
        fig, axes = plt.subplots(3, 4)
        for ax in np.array(axes).flatten():
              next_point,next_pointy = bbopt.step()
              mean, std,_ = bbopt.model.predict(x)    
              plot_1d(ax,x,y,bbopt.observations[0], bbopt.observations[1], mean, std)
              ax.plot(bbopt.best['X'],bbopt.best['Y'],'go')
              ax.plot(next_point,next_pointy,'ro')
              ax.set_ylim([-10,10])                      
              fig.savefig('BNN'+'_'+transfer_func+'_'+'structure'+'_'+str(i)+'.svg')

#if __name__ == '__main__':
plt.show(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)), debug=True)

print('End')