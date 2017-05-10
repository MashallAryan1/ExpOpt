'''
Created on 7/03/2017

@author: mashal
'''
from __future__ import division
from sklearn import preprocessing as pre
import numpy as np
from scipy import stats as st
import deap
from deap import cma
from deap import base
from deap import creator
from deap import tools
from deap import algorithms

class Optimizer(object):
    '''
    optimizer class for expensive optimization
    '''
    def __init__(self, bounding_box, init_observatins, evaluation_function
                 , cma_popsize, cma_max_itr,
                 model_type,  **kwargs):
        '''
        Constructor

        Args:
            bounding_box: 2d numpy array or a list of 2 n dimensional elements 
                  (n is dimensionality of the problem).
                  bounding_box[0] is the lower and bounding_box[1] is
                  the array/list of the upper bound for the features
            init_observatins:  list [feature vector,  observation vector],
                   feature vector is m*n and observation vector is m by 1
                   set of initial observed data to build a primary model
            evaluation_function: function instance
                     the external evaluation function which gets
                     a datum and outputs the related observation/target_value
            cma_popsize:
            cma_max_itr: maximum number of iterations over infill criterion function  
            
            model_type:  string
                         class name of the model to be used as surrogate
            kwargs : set of named parameters and their values for the 
                surrogate model constructor
        '''
        self.bounding_box = bounding_box
        self.observations = init_observatins
        X, Y = self.observations
        # So we do not have a (surrogate)model yet
        self.model = None
        # index of the best observation
        index = np.argmin(Y)
        self.best = {'X': X[index], 'Y': Y[index]}
        # self.standard_scaler = pre.StandardScaler()
        #population size for cma-es( cheap search step) is 10*self.cma_popsize
        self.cma_popsize = cma_popsize
        #maximum number of epoches for cheap search step)
        self.cma_max_itr = cma_max_itr
        #initialize the surrogate model 
        self.init_model(model_type, **kwargs)
        assert self.model != None
        #keep a reference to the external expensive evaluation function 
        self.evaluate = evaluation_function

    def step(self):
        """
        One optimization step
        """
        #update the surrogate with the available data
        self.update_model()
        #       find a new point by doing a cheap optimization
        proposal = self.propose()
        #    evaluate the the proposed point using the expensive objective function
        y = np.array(self.evaluate(proposal))
        #   add the evaluated point to the set of observations
        self.observations_add([proposal, y])
        #   update the best point so far
        if y <= self.best['Y']:
            self.best = {'X': proposal, 'Y': y}
        #   return the result of this step
        return proposal, y

    def init_model(self, model_type, **kwargs):
        #construct the surogate model
        if self.model is None:
            try:
                self.model = model_type(**kwargs)
            except:    
                
                raise ImportError('The specified model class ({} )was not found'.format(str(model_type)))
        self.model.build_model()
        # for _ in range(3):
        #     self.update_model()

    def update_model(self):
        X, Y = self.observations
        self.model.fit(X, Y)
        # self.standard_scaler.fit(np.vstack((X, np.atleast_2d(self.bounding_box)
        #                                     )))
        # Xtr = self.standard_scaler.transform(X)
        # self.model.fit(Xtr, Y)

    def propose(self):
        # cma-es considers variables to be in [0,10] 
        # so we need to transform the results
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Individual", np.ndarray, fitness=creator.FitnessMin)

        lower = np.array(self.bounding_box[0])
        upper = np.array(self.bounding_box[1])
        toolbox = base.Toolbox()
        # to stay inside the boundary[a,b]: a + (b-a) * (1 - cos(pi * x / 10)) / 2
        toolbox.register("evaluate", (lambda x: (-self.expected_improvement(lower + (upper - lower) * (1 - np.cos(np.pi * x / 10)) / 2),)))

        strategy = cma.Strategy(centroid=np.array([5.0]*self.dimensionality), sigma=5.0, lambda_=self.cma_popsize*self.dimensionality)
        toolbox.register("generate", strategy.generate, creator.Individual)
        toolbox.register("update", strategy.update)

        hof = tools.HallOfFame(1, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean)
        stats.register("std", np.std)
        stats.register("min", np.min)
        stats.register("max", np.max)

        algorithms.eaGenerateUpdate(toolbox, ngen=self.cma_max_itr, stats=stats, halloffame=hof)
        bestfound =np.array(lower + (upper - lower) * (1 - np.cos(np.pi * hof[0] / 10)) / 2)
        return bestfound 
        
        

    def expected_improvement(self, x):
        mean, std, _ = self.model.predict(x)
        if std == 0:
            return (self.best['Y'] - mean)
        u = (self.best['Y'] - mean)/std
        EI = u*st.norm.cdf(u) + st.norm.pdf(u)
#        print EI
        #EI[np.where(std == 0)] = np.maximum(self.best['Y'] - mean, 0)[np.where(std == 0)]
        # if EI.shape[0] == 1:
#            return EI[0]
        return EI

    @property
    def observations(self):
        return self._observations['X'], self._observations['Y']

    @observations.setter
    def observations(self, value):
        self._observations = {'X': np.atleast_2d(value[0]), 'Y': np.atleast_2d(value[1])}
        
    @property    
    def dimensionality(self):
        return np.array(self.bounding_box).shape[1]
        
    def observations_add(self, value):
        X = np.atleast_2d(value[0])
        Y = np.atleast_2d(value[1])
        self._observations['X'] = np.vstack(( self._observations['X'], X))
        self._observations['Y'] = np.vstack(( self._observations['Y'], Y))


    