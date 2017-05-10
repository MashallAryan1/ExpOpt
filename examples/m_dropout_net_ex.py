from models.feed_forward_net import *
from mashall.functions import *
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
mlp = MlpRegressor( structure =(1,500,100,100,500,1),dropout_rate =0.5, dropout_type = 'Gaussian' #{ "zero_one" or "Gaussian"}
                    ,  use_dropout_ontest = True, num_training_steps=400000, learning_rate =0.5
                    , ensemble_size =50, batch_size=50, standardize=True)

def g(x):
    return x*np.sin(x)

num_points = 50   

xt=np.linspace(-20,20,num_points).reshape(num_points,1)
x= np.linspace(-20,20,num_points*10).reshape(num_points*10,1)
yt=g(xt)
y = g(x)
assert y.shape == x.shape

mlp.fit(xt,yt)
mean, std, ys = mlp.predict(x)
fig, ax = plt.subplots(1, 1)
plot_1d(ax,x,y,xt,yt,mean,std)
fig.savefig('display.svg')
#error history.losses
fig, ax = plt.subplots(1, 1)
ax.plot(mlp.history.losses)
fig.savefig('error.svg')
# #if __name__ == '__main__':
plt.show(host=os.getenv('IP', '0.0.0.0'),port=int(os.getenv('PORT', 8080)), debug=True)