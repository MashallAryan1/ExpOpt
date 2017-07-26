from __future__ import absolute_import, division, print_function

from sklearn import preprocessing as pre
import edward as ed
from edward.models import Normal
import tensorflow as tf
import numpy as np

#To do:  transfer shared methods and properties to an upper class

class Bayesian_neural_net(object):


    def __init__(self,**kwargs):
        """

        Args:

            'structure': a tuple representing the architecture of the neural net. e.g (3,4,4,1) is a ann with 3 input
                     nodes and 2 hidden layers each containing 4 units and a single output
            'transfer_funcs': activation function(s) of the hidden layers: can be tuple if each layer has
                     a different activation function type
            'learning_rate' : learning rate for the optimization algorithm
            'num_training_steps':  inr
                            number of training steps
            'ensemble_size': int
                            using dropout during testing/prediction  prediction is performed using an ensemble of ANNs  with its size set to ensemble_size
            'batch_size':   int
                        batch size if the training set is large
            'standardize': boolean (default=True)
                          a flag to indicate if the feature vector should be standardized: (zero mean unit variance)
            'noise_std'  :real
                          Standard deviation of output noise
        """
        #initialization of the properties
        self.transfer_funcs={'relu':tf.nn.relu,'tanh':tf.nn.tanh,'sigmoid':tf.nn.sigmoid}
        self.structure = kwargs.get('structure',None)
        self.transfer_func =self.transfer_funcs[kwargs.get('transfer_func', 'relu')]
        #self.learning_rate = kwargs.get('learning_rate', 0.001)
        self.num_training_steps = kwargs.get('num_training_steps', 1000)
        self.ensemble_size = kwargs.get('ensemble_size', 500)
        self.standardize = kwargs.get('standardize', True)
        self.scaler = None
        #noise std(should be removed later)
        self.noise_std =  kwargs.get('noise_std',0.01)
        # dictionary of priors and their corresponding variational distributions
        self.params_dict = None
        # input data placeholder
        self.X = None
        # likelihood variable
        self.pred = None
        self.build_model()

    def build_model(self):
        """
        builds the structure of the Neural net, defines the priors and variatinal distributions

        """
        # put priors over weights
        weights = []
        biases = []
        q_w = []
        q_b = []
        #for each layer
        for index in range(len(self.structure)-1):
            # define priors over weights and biases
            #layer_weights = Normal(loc=tf.zeros([self.structure[index], self.structure[index+1]]), scale=tf.ones([self.structure[index], self.structure[index+1]]))
            layer_weights = Normal(loc=tf.zeros([self.structure[index],self.structure[index+1]]), scale=tf.ones([self.structure[index],self.structure[index+1]]))
            weights.append(layer_weights)
            layer_biases = Normal(loc=tf.zeros([self.structure[index+1]]), scale=tf.ones([self.structure[index+1]]))
            biases.append(layer_biases)
            # define the corresponding variational distributions
            q_layer_weights = Normal(loc=tf.Variable(tf.random_normal([self.structure[index],self.structure[index+1]])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.structure[index],self.structure[index+1]]))))
            q_w.append(q_layer_weights  )
            q_layer_biases = Normal(loc=tf.Variable(tf.random_normal([self.structure[index+1]])), scale=tf.nn.softplus(tf.Variable(tf.random_normal([self.structure[index+1]]))))
            q_b.append(q_layer_biases)
        # define placeholders   for  input data
        self.X = tf.placeholder(tf.float32,[None,self.structure[0]])
        # define the likelihood
        self.pred = Normal(loc=self.network(self.X,weights,biases) , scale=self.noise_std*tf.ones([1]))
        # input dictionary for the inference method
        self.params_dict= {}
        for i in range(len(weights)):
            self.params_dict[weights[i]] = q_w[i]
            self.params_dict[biases[i]] = q_b[i]



    def network(self, x, weights, biases):
        #build the network-> first layer
        h = self.transfer_func(tf.matmul( x ,weights[0]) + biases[0])
        #build the network-> hidden layers
        for i in range(1,len( weights) - 1):
            h = self.transfer_func(tf.matmul(h, weights[i]) + biases[i])
        #build the network-> last layer
        return tf.matmul(h, weights[-1]) + biases[-1]

    def fit(self, xt, yt):
        """
        fit the model to the training set
        """
        x = xt
        if self.standardize:
            #standardize the input
            self.scaler = pre.StandardScaler().fit(xt)
            x = self.scaler.transform(xt)
        y = np.atleast_2d(yt)
        assert xt.shape[0] == yt.shape[0]
        # specify the number of observations
        n_training = xt.shape[0]

        #self.build_model(n_training)
        inference = ed.KLqp(self.params_dict, data={self.X:x, self.pred:y} )
        inference.run(n_samples=30, n_iter=self.num_training_steps)
        #self.model.fit(x,yt, batch_size= self.batch_size, epochs=self.num_training_steps)
#        scores = self.model.evaluate(xt, yt)


    def single_prediction(self, xt):
        #predictive posterior distribution
        predictive_dist = ed.copy(self.pred, self.params_dict )
        x = xt
        if self.standardize:
            x = self.scaler.transform(xt)
        return ed.get_session().run( predictive_dist,{self.X:x})

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

