from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
matplotlib.use('Agg')
import os
from bboptim.optimization import *
from models.dropout_net import SDropoutRegressor
import unittest
import numpy as np
from mashall.functions import *
from matplotlib import pyplot as plt
np.random.seed(124)
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

model_type = SDropoutRegressor


#parameter configurations:
dropout_types = ['Gaussian','zero_one']
dropout_rates =[ 0.05, 0.1, 0.2, 0.5]
#hiddenlayerCount = [ 5, 7]

transfer_funcs = ['tanh',]
structures = [   (1,)+(1024,)*4+(1,),  (1,)+(1024,)*6+(1,)]
#print structures
j=0
for dropout_type in dropout_types:
    for dropout_rate in dropout_rates:
        for transfer_func in transfer_funcs:
            for i,structure in enumerate(structures):
                # for the description of paramete3rs goto "feed_forward_net.py" and optimization.py
                bbopt = Optimizer( bounding_box=box, init_observatins=None#[xt,yt]
                                    , evaluation_function = g#,  n_init_random_samples = 0
                                    , cma_popsize=20, cma_max_itr=50, model_type=SDropoutRegressor
                                    , structure=structure
                                    , dropout_rate=dropout_rate
                                    , dropout_type=dropout_type #{ "zero_one" or "Gaussian"}
                                    , use_dropout_ontest = True
                                    , use_batch_normalization=True
                                    , use_early_stopping=True
                                    , transfer_func=transfer_func
                                    , num_training_steps=30000
                                    , learning_rate=1.0e-3
                                    , ensemble_size=60
                                    , batch_size=32
                                    , weight_decay=0.001
                                    , standardize=True)
                bbopt.search(n_init_steps=4, n_steps=5)
                fig, axes = plt.subplots(3, 4)
                for ax in np.array(axes).flatten():
                    next_point,next_pointy = bbopt.step()
                    mean, std,_ = bbopt.model.predict(x)
                    plot_1d(ax,x,y,bbopt.observations[0], bbopt.observations[1], mean, std)
                    ax.plot(bbopt.best['X'],bbopt.best['Y'],'go')
                    ax.plot(next_point,next_pointy,'ro')
                    ax.set_ylim([-12,12])
                    fig1, ax1 = plt.subplots(1, 1)
                    ax1.plot(bbopt.model.history.losses)
                    j+=1
                    fig1.savefig('error_{}.png'.format(j))
                    fig.savefig(dropout_type+'_'+str(dropout_rate)+'_'+transfer_func+'_'+str(i)+'.png')

#if __name__ == '__main__':
#plt.show(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)), debug=True)
print('done')
