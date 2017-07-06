'''
Created on 7/03/2017

@author: mashal
'''
from __future__ import absolute_import, division, print_function, unicode_literals
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
            n_init_random_samples: int
                                  number of initial random samples after which BO is applied

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
        self._observations = {}
        self.bounding_box = bounding_box

        # So we do not have a (surrogate)model yet
        self.model = None
        # index of the best observation
        self.best = None
        # self.standard_scaler = pre.StandardScaler()
        #population size for cma-es( cheap search step) is 10*self.cma_popsize
        self.cma_popsize = cma_popsize
        #maximum number of epoches for cheap search step)
        self.cma_max_itr = cma_max_itr

        self.init_XY(init_observatins)

        #initialize the surrogate model
        self.init_model(model_type, **kwargs)

        assert self.model != None
        #keep a reference to the external expensive evaluation function
        self.evaluate = evaluation_function


    def init_XY(self,init_observatins):
        if not init_observatins:
            return
        X, Y = init_observatins
        if X  is not None:
            self.observations = init_observatins
            # index of the best observation
            index = np.argmin(Y)
            self.best = {'X': X[index], 'Y': Y[index]}

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
        #print self.best['X']
        acqusition_function = (lambda x: ( self.expected_improvement(self.best['X'])-self.expected_improvement(lower + (upper - lower) * (1 - np.cos(np.pi * x / 10)) / 2),))
        #acqusition_function = (lambda x: ( -self.expected_improvement(lower + (upper - lower) * (1 - np.cos(np.pi * x / 10)) / 2),))

        #toolbox.register("evaluate", (lambda x: (-self.expected_improvement(lower + (upper - lower) * (1 - np.cos(np.pi * x / 10)) / 2),)))
        toolbox.register("evaluate", acqusition_function)

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
        #bestfound =np.array(hof[0])
        return bestfound



    def expected_improvement(self, x):
        mean, std, _ = self.model.predict(x)
        if isinstance(std, np.ndarray):
            std[std==0] = 1e-6
        elif std == 0:
            std = 1e-6
        normal  = st.norm(loc=mean,scale=std)
        return (self.best['Y'] - mean)*normal.cdf(self.best['Y'])+std**2*normal.pdf(self.best['Y'])

#         u = (self.best['Y'] - mean)/std
#         EI = u*st.norm.cdf(u) + st.norm.pdf(u)
#        print EI
        #EI[np.where(std == 0)] = np.maximum(self.best['Y'] - mean, 0)[np.where(std == 0)]
        # if EI.shape[0] == 1:
#            return EI[0]
#         return EI

    def random_step(self):
        """
        One random optimization step
        """
        lower,upper = self.bounding_box[0], self.bounding_box[1]
        rnd = np.random.rand(lower.shape[0])
        proposal = lower + rnd*(upper-lower)
        y = np.array(self.evaluate(proposal))
        self.observations_add([proposal, y])
        self.update_best({'X': proposal, 'Y': y})

#         if self.best is None  or self.best['Y'] is None:
#             self.best = {'X': proposal, 'Y': y}
#         elif y <= self.best['Y']:
#             self.best = {'X': proposal, 'Y': y}
        return proposal, y



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
        self.update_best({'X': proposal, 'Y': y})
        #         if y <= self.best['Y']:
        #             self.best = {'X': proposal, 'Y': y}
        #   return the result of this step
        return proposal, y

    def search(self, n_init_steps = 0, n_steps = 100):
        i=0
        while i<n_init_steps:
            self.random_step()
            i+=1
            print ('initial random search step {}'.format( i))
        i = 0
        while i<n_steps:
            self.step()
            i+=1
            print ('main optimization loop step {}'.format( i))


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
        if  not self._observations:
            self.observations = value
            return

        self._observations['X'] = np.vstack(( self._observations['X'], X))
        self._observations['Y'] = np.vstack(( self._observations['Y'], Y))

    def update_best(self,xy_pair):
        if self.best is None or xy_pair['Y'] < self.best['Y']:
            self.best = xy_pair



