from __future__ import division

from sklearn import preprocessing as pre
import keras
from keras.layers.core import Lambda
from keras import backend as K
from keras.models import Sequential
from keras.layers import Activation, Dropout, BatchNormalization, Dense
from keras.layers.noise import GaussianDropout
import numpy as np
from mashall.utils import *
class MlpRegressor(object):
     

    def __init__(self,**kwargs):
        """
        
        Args:
        
            'structure': a tuple representing the architecture of the neural net. e.g (3,4,4,1) is a ann with 3 input
                     nodes and 2 hidden layers each containing 4 units and a single output
            'transfer_funcs': activation function(s) of the hidden layers: can be tuple if each layer has
                     a different activation function type
            'learning_rate' : learning rate for the optimization algorithm  
            'dropout_rate': float (0,1)
                            dropout rate (1- keep_prob) used for dropout while training
            'dropout_type': { "zero_one" or "Gaussian"}
                            type of dropout used 
            'use_dropout_ontest' : booelan
                                    A flag indicating the presence of dropout during the prediction/test
            'num_training_steps':  inr
                            number of training steps     
            'ensemble_size': int 
                            using dropout during testing/prediction  prediction is performed using an ensemble of ANNs  with its size set to ensemble_size
            'batch_size':   int
                        batch size if the training set is large    
            'standardize': boolean (default=True)
                          a flag to indicate if the feature vector should be standardized: (zero mean unit variance)
            'use_batch_normalization': boolean (default=True)
                                        indicates if batch normalization is used
            'use_early_stopping': Boolean (default=True)                            
                                    Will stop the training after a specific number of no improvement in 'loss' value
                                    (curently parameters are hard coded)
            'use_batch_normalization':       Boolean (default=True)                    
        """
        #initialization of the properties
        self.structure = kwargs.get('structure',None)
        self.transfer_func = kwargs.get('transfer_func', 'relu')
        self.learning_rate = kwargs.get('learning_rate', 1.0e-4)
        self.dropout_rate = kwargs.get('dropout_rate', 0.3 )
        self.dropout_type = kwargs.get('dropout_type', 'zero_one')
        self.use_dropout_ontest = kwargs.get('use_dropout_ontest', True )
        self.num_training_steps = kwargs.get('num_training_steps', 1000)            
        self.ensemble_size = kwargs.get('ensemble_size', 50)            
        self.batch_size = kwargs.get('batch_size', 1)
        self.standardize = kwargs.get('standardize', True)
        self.use_early_stopping = kwargs.get('use_early_stopping', True)
        self.use_batch_normalization =  kwargs.get('use_batch_normalization', True)
        self.scaler = None
        # self.X, self.Y, self.regularization_factor, self.keep_prop = None, None, None, None
        # self.prediction, self.loss, self.step = None, None, None
        self.history=None
        self.build_model()

    def build_model(self):
        """
        builds the structure of the Neural net
                                
        """
        #sequentioal model
        self.model = Sequential()
    
        #first layer 

        self.model.add(Dense(self.structure[1],kernel_initializer='normal',input_dim=self.structure[0]))
        for layer in range(2,self.n_layers):
            # add  batch normalization 
            if self.use_batch_normalization:
                self.model.add(BatchNormalization())
            # activation function    
            self.model.add(Activation(self.transfer_func))
            # drop out 
            self.model.add({'zero_one':{False:Dropout(self.dropout_rate),# self.use_dropout_ontest == False
                             True:Lambda(lambda x: K.dropout(x, level=self.dropout_rate))},#self.use_dropout_ontest == True
                             'Gaussian':{False:GaussianDropout(self.dropout_rate),# self.use_dropout_ontest == False
                             True:Lambda(lambda x:   x*K.random_normal(shape=K.shape(x),
                                                mean=1.0,
                                                stddev=np.sqrt(self.dropout_rate / (1.0 - self.dropout_rate))))}}[self.dropout_type][ self.use_dropout_ontest])
                                                
            # next dense layer    
            self.model.add(Dense(self.structure[layer],kernel_initializer='normal'))
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)    #lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0

    	self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        
    
    def fit(self, xt, yt):
        """
        fit the model to the training set 
        """
        optimizer = keras.optimizers.Adam(lr=self.learning_rate)    #defaut: lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0
    	self.model.compile(loss='mean_squared_error', optimizer=optimizer)
        x=xt
        if self.standardize:
            #standardize the input
            self.scaler = pre.StandardScaler().fit(xt)
            x = self.scaler.transform(xt)
        y = np.atleast_2d(yt)
        assert xt.shape[0] == yt.shape[0]
        callbacks = [keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.8, patience=50, min_lr=1.0e-5)]
        self.history = LossHistory()
        callbacks += [self.history]
        if self.use_early_stopping:
            callbacks+=[keras.callbacks.EarlyStopping(monitor='loss', min_delta=1.0e-5, patience=5000, verbose=0, mode='auto')]
        self.model.fit(x,yt, batch_size= self.batch_size, epochs=self.num_training_steps, callbacks=callbacks)
#        scores = self.model.evaluate(xt, yt)

        

    def single_prediction(self, x):
        if self.standardize:
            return self.model.predict(self.scaler.transform(x))
        else:    
            return self.model.predict(x)

    def predict(self, xt):
        
        xs =  np.atleast_2d(xt)
        ys = []
        for _ in range(self.ensemble_size):
            prediction= self.single_prediction(xs)
            ys.append(prediction)
        ys = np.squeeze( ys )
        std =  np.std( ys,axis=0 )
        #var = np.var(ys,axis =0 )+ self.sess.run(self.tau)**-1
        mean = np.average( ys,axis=0 )
        return mean, std, ys

    """
    Todo: remove the summary and writers
    """
    @property
    def n_layers(self):
        return len(self.structure)
            

    @property
    def structure(self):
        return self._structure
    
    @structure.setter
    def structure(self, value):
        if not (value or isinstance(value, tuple)):
            raise ValueError('structure parameter must be initial4,3,7ized to a tuple e.g. (3,2,1).')
        elif len(value) < 3:
            raise ValueError('structure must a least contain 3 non zero integer items.')
        else:
            for item in value:
                if not isinstance(item, int) or (isinstance(item, str) and not item.isdigit()):
                    raise ValueError('structure must contain non zero integer items.')
        self._structure =  value

    @property  
    def transfer_func(self):
        return self._transfer_func
    
    @transfer_func.setter
    def transfer_func(self, value):
        self._transfer_func = value

    @property
    def learning_rate(self):
        return self._learning_rate
    @learning_rate.setter            
    def learning_rate(self, value):
        self._learning_rate = value
    
    @property
    def dropout_rate(self):
        return self._dropout_rate
    
    @dropout_rate.setter
    def dropout_rate(self, value):
        self._dropout_rate = value

    @property
    def use_dropout_ontest(self):
        return self._use_dropout_ontest
    
    @use_dropout_ontest.setter
    def use_dropout_ontest(self, value):
        self._use_dropout_ontest = value



    @property     
    def num_training_steps(self):
        return self._num_training_steps
    @num_training_steps.setter
    def num_training_steps(self, value):
        self._num_training_steps = value


    @property     
    def ensemble_size(self):
        return self._ensemble_size
    
    @ensemble_size.setter
    def ensemble_size(self, value):
        self._ensemble_size = value

