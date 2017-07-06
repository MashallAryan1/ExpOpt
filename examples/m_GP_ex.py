from __future__ import absolute_import, division, print_function, unicode_literals
from models.gaussian_process import *
from mashall.functions import *
import numpy as np
import os
import matplotlib
import edward as ed
ed.set_seed(42)
matplotlib.use('Agg')
import matplotlib.pyplot as plt
model = GPRegressor( noise_std = 0.1
                    , num_training_steps=2000
                    , standardize=True)

def build_toy_dataset(D, left1=-10.0,right1=1.6,left2=6,right2=10, N=40, noise_std=1.0e-5):
  #D = 1
  X = np.concatenate([np.linspace(left1, right1, num=N*D / 2),
                      np.linspace(left2, right2, num=N*D / 2)]).reshape(N,D)
  y = np.sum(np.cos(X),axis=1) + np.random.normal(0, noise_std, size=N)
  X = X.reshape((N, D))
  return X, y

# DATA
N =400 # number of data points
D = 1 # number of features
xt, yt = build_toy_dataset(D, left1=-10,right1=0,left2=0,right2=10, N=N, noise_std=1.0e-5)

yt = yt.reshape(N,1)
x = np.linspace(-11, 11, num=2000*D).reshape(2000,D)
y = np.cos(x)#+np.random.normal(0, 1.0e-5, size=400)

# def g(x):
#     return x*np.sin(x)

#num_points = 10

# xt=np.linspace(-20,20,num_points).reshape(num_points,1)
# x= np.linspace(-20,20,num_points*10).reshape(num_points*10,1)
#yt=g(xt)
#y = g(x)


model.fit(xt,yt)
mean, std, ys = model.predict(x)

fig, ax = plt.subplots(1, 1)
plot_1d(ax,x,y,xt,yt,mean,std)

fig.savefig('display.svg')
print('Done')
#error history.losses
#fig, ax = plt.subplots(1, 1)
#ax.plot(model.history.losses)
#fig.savefig('error.svg')
# #if __name__ == '__main__':
plt.show(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)), debug=True)