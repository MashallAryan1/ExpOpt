'''
Created on 6/04/2017

@author: mashal
'''
import numpy as np

def plot_1d(ax,x,y,xt,yt,mean,std):
    """
    plots x agains y as a trend and xt against yt as points where mean and std 
    are supposed to be the mean and std of the prediction of a regressor over x
    """
    ax.plot(x, mean, 'r')
    #ax.plot(x, y_hat, 'r')
    ax.plot(x, y, 'b')
    ax.plot(xt, yt, 'o')
    x1 = np.squeeze(x)
    for i in range(1, 10):
        ax.fill(np.concatenate([x1, x1[::-1]]),
                 np.concatenate([mean - i*0.3 * std,
                                 (mean + i *0.3 * std)[::-1]]),
                 alpha=0.08, fc='b', ec='None', label='2*Sigma')

class exhaustive_search(object):
    """
    exhaustive search for 1d space
    """
    def __init__(self,lowerbound, upperbound,n_points):
        self.bounds = [lowerbound, upperbound]
        self.n_points = n_points
        
    def optimize(self,func):    
        grid_points = np.linspace(lowerbound[0], upperbound[0],  self.n_points).reshape( self.n_points,1)
        values = func(grid_points)
        best_index= np.argmin(values)
         
        return (grid_points[best_index],values[best_index])

    
        
    