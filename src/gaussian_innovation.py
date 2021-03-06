from sys import exit
import numpy as np
from regime_switching_ARM import RSARM

############################################################################
## @class Gaussian
#  Regime switching linear auto-regressive model with gaussian innovation
#
class Gaussian_X(RSARM):

    ## @fn      updated 21/06/13
    #  @brief
    #  
    #  @param order Autoregressive process order
    #  @param nb_regime Number of regimes
    #  @param data Vector of length S, where S is the numbers of observed 
    #  sequences, data[s] is s^th sequence, is a column vector
    #  T_s x 1.
    #  @param initial_values Matrix orderx1, initial values of X.
    #  @param method Initialization method to be considered. Can take two
    #   values :
    #       * "rand" for random initialization. Default value
    #       * "datadriven" 
    #   
    def __init__(self, order, nb_regime, data, initial_values, states, \
                 method="rand"):
        
        #assertion
        assert (order >= 0)
        assert (nb_regime > 0)
        
        self.innovation = "gaussian"
        self.order = order
        self.nb_regime = nb_regime
        self.data = data
        self.initial_values = initial_values
        
        #initial law parameters
        self.psi = {}
        self.psi["means"] = np.zeros(shape=self.order, dtype=np.float64)
        self.psi["covar"] = np.identity(n=self.order, dtype=np.float64)
        
        #-----------LAR parameters' initialization
        #random initialization: used for CMAPSS datasets and synthetic data
        if(method == "rand"):           
            self.intercept = np.zeros(dtype=np.float64, shape=self.nb_regime)
            self.coefficients = np.random.uniform(-1.0, 1.0, (self.order, self.nb_regime))   
            self.sigma = np.random.uniform(0.2, 4, self.nb_regime)
            
        elif(method == "datadriven"):  
            #--compute the unconditional mean and covariance of data
            unc_mean = 0.0
            unc_std = 0.0
            S = len(self.data) 
            for i in range(S):
                unc_mean = unc_mean + np.mean(self.data[i])
                unc_std = unc_std + np.std(self.data[i])
                
            unc_mean = unc_mean / S
            unc_std = unc_std / S
            
            #--initialization
            self.coefficients = np.zeros(dtype=np.float64, \
                                     shape=(self.order, self.nb_regime)) 
            self.sigma = unc_std * np.ones(dtype=np.float64, shape=self.nb_regime) 
            
            two_std = 2*unc_std
            self.intercept = np.random.uniform(unc_mean-two_std, \
                                               unc_mean+two_std, \
                                               self.nb_regime)
            
        else:
            print("ERROR: in class gaussian, unkown initialization method!\n")
            exit(1)
       
        return         
        

    ## @fn
    #  @brief
    #  @todo check model dimensions
    #
    #  @param intercept
    #  @param coefficients
    #  @param sigma
    #
    def set_parameters(self, intercept, coefficients, sigma):
        self.intercept = intercept
        self.coefficients = coefficients  
        self.sigma = sigma
        
        return 
