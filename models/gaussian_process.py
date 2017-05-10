import GPflow
import numpy as np
import copy
from sklearn import preprocessing as pre
#TODO: rafactor to include all possible GP models
class GPRegressor(object):

    def __init__(self,**kwargs):
        """
        
        Args:
        
            'num_training_steps':  int
                            number of training steps    
            'input_dim'             :  int
                                   dimensionality of the input
            'batch_size':   int
                        batch size if the training set is large    
            'standardize': boolean (default=True)
                          a flag to indicate if the feature vector should be standardized: (zero mean unit variance)
            'kernel'    :string, a kernel class, or a list of kernel classes(default = 'Matern52')
                         GP kernel from the list of available GPflow kernels. 
                         Can contain the name(s) of kernel(s) can be provided as a string(comma delimited in case of multiple kernels),
                         or the kernel class(one kernel) or a list of multiple kernel classes 
            'ARD'       :boolean(default = True)             
                         the ARD option for the stationary Kernels
            'meanf'     : string(default = Zero)        
                          name of the mean function from list of available mean function in GPflow.    
                          currently only supports Zero
            'mean_params':tuple
                          tuple of parameters values for the mean function if it accepts any    
                          
            'noise_std'  :real
                          standard deviation of likelihood noise
                          
                         
        """
        #initialization of the properties
        self.num_training_steps = kwargs.get('num_training_steps', 1000)    
        # self.input_dim = kwargs.get('input_dim', None)
        self.batch_size = kwargs.get('batch_size', 1)
        self.standardize = kwargs.get('standardize', True)
        
        self.kernel = getattr(GPflow.kernels, kwargs.get('kernel', 'Matern52') )
        self.ARD = kwargs.get('ARD',True)  
        self.meanf = kwargs.get('meanf','Zero') 
        self.mean_params = kwargs.get('mean_params',None)  
        self.model_type =  kwargs.get('model_type',GPflow.gpr.GPR)  
        self.noise_std =  kwargs.get('noise_std',0.1)
        self.scaler = None
        # self.X, self.Y, self.regularization_factor, self.keep_prop = None, None, None, None
        # self.prediction, self.loss, self.step = None, None, None
        self.build_model()
    
    def build_model(self):
        """
        
                                
        """
        pass


    
    def fit(self, xt, yt):
        """
        fit the model to the training set 
        """
        x=xt
        if self.standardize:
            #standardize the input
            self.scaler = pre.StandardScaler().fit(xt)
            x = self.scaler.transform(xt)
        y = np.atleast_2d(yt)
        assert xt.shape[0] == yt.shape[0]
        #define the kernel
        k = GPflow.kernels.Matern52(xt.shape[1], ARD = self.ARD)
        k.lengthscales.transform = GPflow.transforms.Logistic(1e-5,5)
        k.variance.transform = GPflow.transforms.Logistic(0.5, 1.5)
        #define the mean
        self.meanf = GPflow.mean_functions.Zero
        #GP model
        self.model = GPflow.gpr.GPR(x, y, k, self.meanf())
        #likelihood
        self.model.likelihood.variance.transform = GPflow.transforms.Logistic(1e-5,0.1)
        self.model.likelihood.variance = self.noise_std**2     
        #fit
        self.model.optimize(maxiter=self.num_training_steps)     
        print(self.model)

        

    def single_prediction(self, xt):
        x = xt
        if self.standardize:
            x= self.scaler.transform(x)
        mean, var = self.model.predict_y(x)    
        return mean,  np.sqrt(var)   

    def predict(self, xt):
        
        xs =  np.atleast_2d(xt)
        mean, std = self.single_prediction(xs)   
        return mean, std, []

    """
    Todo: remove the summary and writers
    """

    @property
    def kernel(self):
        
        return self._kernel
        
    @kernel.setter    
    def kernel(self, value):
        if isinstance(value, basestring):
            #if kernel name(s) is(are) passed as a (comma delimited)string
            kernel_names= value.split()
            self._kernel = [getattr(GPflow.kernels, kername) for kername in kernel_names]
        elif isinstance(value,list):    
            #list of kernels kernel classes
            self._kernel = [copy.copy(kernel) for kernel in value]
        elif isinstance(value(1), GPflow.kernels.Kern):
            self._kernel = [copy.copy(value)]
            

    @property
    def meanf(self):
        return self._meanf
    @meanf.setter    
    def meanf(self, value):
        
        self._meanf = value#getattr(GPflow.mean_functions, value)
    