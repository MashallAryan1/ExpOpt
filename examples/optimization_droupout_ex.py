from __future__ import division
import matplotlib
matplotlib.use('Agg')
import os
from bboptim.optimization import *
from models.feed_forward_net import MlpRegressor
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
n_training = 2
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

model_type = MlpRegressor


#parameter configurations:
dropout_types = ['zero_one','Gaussian']
dropout_rates =[0.1, 0.05]
#hiddenlayerCount = [ 5, 7]

transfer_funcs = ['relu','tanh']
structures = [ (1,200,200,200,1),(1,300,300,1),(1,250,100,250,1),(1,295,10,295,1),(1,100,100,100,100,100,100,1)][::-1]
j=0
for dropout_type in dropout_types:
    for dropout_rate in dropout_rates:
        for transfer_func in transfer_funcs:
            for i,structure in enumerate(structures):
                # for the description of parameters goto "feed_forward_net.py" and optimization.py
                bbopt = Optimizer( bounding_box=box, init_observatins=[xt,yt]
                                    , evaluation_function = g
                                    , cma_popsize=20, cma_max_itr=20, model_type=MlpRegressor
                                    , structure=structure
                                    , dropout_rate=dropout_rate
                                    , dropout_type=dropout_type #{ "zero_one" or "Gaussian"}
                                    , use_dropout_ontest = True
                                    , use_batch_normalization=True
                                    , use_early_stopping=True
                                    , transfer_func=transfer_func
                                    , num_training_steps=500000
                                    , learning_rate=1.0e-1
                                    , ensemble_size=30
                                    , batch_size=32
                                    , weight_decay=1.0e-6
                                    , standardize=True)
                fig, axes = plt.subplots(3, 4)
                for ax in np.array(axes).flatten():
                      next_point,next_pointy = bbopt.step()
                      mean, std,_ = bbopt.model.predict(x)
                      plot_1d(ax,x,y,bbopt.observations[0], bbopt.observations[1], mean, std)
                      ax.plot(bbopt.best['X'],bbopt.best['Y'],'go')
                      ax.plot(next_point,next_pointy,'ro')
                      ax.set_ylim([-12,12])
                      fig.savefig(dropout_type+'_'+str(dropout_rate)+'_'+transfer_func+'_'+str(i)+'.svg')
                      fig1, ax1 = plt.subplots(1, 1)
                      ax1.plot(bbopt.model.history.losses)
                      j+=1
                      fig1.savefig('error_{}.svg'.format(j))

#if __name__ == '__main__':
plt.show(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)), debug=True)

