from __future__ import absolute_import, division, print_function, unicode_literals
import matplotlib
#matplotlib.use('Agg')
import os
from bboptim.optimization import *
from models.gaussian_process import GPRegressor
import unittest
import numpy as np
from mashall.functions import *
from matplotlib import pyplot as plt
np.random.seed(123)
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
n_test = 1000

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
noise_std = 0.1



bbopt = Optimizer( bounding_box=box, init_observatins=[xt,yt]
                    , evaluation_function = g
                    , cma_popsize=20, cma_max_itr=20, model_type=GPRegressor
                    , num_training_steps=500
                    , standardize=True, noise_std = noise_std)
# for _ in range(20):
#      bbopt.step()
fig, axes = plt.subplots(3, 4)
fig2,axes2= plt.subplots(3, 4)
axes2f = np.array(axes2).flatten()
i=0
for ax in np.array(axes).flatten():
      next_point,next_pointy = bbopt.step()
      print('-------------------------')
      mean, std,_ = bbopt.model.predict(x)
      axes2f[i].plot(x,np.array([bbopt.expected_improvement(xx[0]).squeeze() for xx in x]))
      i+=1
      plot_1d(ax,x,y,bbopt.observations[0], bbopt.observations[1], mean, std)
      ax.plot(bbopt.best['X'],bbopt.best['Y'],'go')
      ax.plot(next_point,next_pointy,'ro')
      #ax.set_ylim([-10,10])
print('done')
      #fig.savefig('EGO.svg')
plt.show(True)
#if __name__ == '__main__':
#plt.show(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)), debug=True)

