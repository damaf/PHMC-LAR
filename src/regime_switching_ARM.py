#NB: numba is a parallelization library. It does support neither scipy library
#nor class members also named class functions.
#import numba as nb

import numpy as np
import math
from scipy import optimize
from scipy.stats import norm, multivariate_normal
import concurrent.futures
#####################################################################################
##  @class LARM
#   Regime switching auto-regressive model
#
class RSARM():

    # Class attributes
               
    def __init__(self):
        ## @brief Distribution of residuals, can be any distribution 
        #  parameterized by its mean and variance, e.g. gaussian, gamma, log-normal ...
        #  
        self.innovation = ""
        
        ## @brief Order of the autoregressive process X.
        #  
        self.order = 0
        
        ## @brief The number of possible execution regimes of X.
        #  
        self.nb_regime = 0
        
        ## @brief Auto-regressive coefficients: \n
        #  order x nb_regime array.
        #  Each regime has its own auto-regressive coefficients.
        #  coefficients[i,k] is the coefficient associated with lagged value 
        #  X_{t-{i+1}} under state k.
        #
        self.coefficients = {}
        
        ## @brief Standard deviation:  \n
        #  Array of length nb_regime.
        # 
        self.sigma = {}
        
        ## @brief Intercept parameter : \n
        #  Array of length nb_regime. 
        # 
        self.intercept = {}
        
        ## @brief List of length S, where S is the number of observed
        #  time series. data[s], the s^th time series, is a column vector
        #  T_s x 1 where T_s denotes its size, starting at timestep
        #  t = order + 1 included.
        #
        self.data = {}
        
        ## @brief List of length S, where S is the number of observed
        #  time series. initial_values[s] is a column vector orderx1 of 
        #  initial values associated with the s^th time series.
        #
        self.initial_values = {}
        
        ## @brief parameters of initial_values law. A multi-variate normal 
        #   distribution of dimension self.order is considered.
        #   Dictionnary having two entries:
        #    * entry "means" self.order 1-D array of means.
        #    * entry "covar" self.order x self.order variance-covariance matrix.
        #
        self.psi = {}
 
       
    ## @fn
    #
    def __str__(self):
        return "innovation " + self.innovation   
 
    
    ## @fn
    #  @brief compute the MLE estimator of the initial law which is a 
    #   multi-variate normal distribution
    #
    def estimate_psi_MLE(self):
        
        #----------no autoregressive dynamics
        if(self.order == 0):
            return 
        
        #----------self.order autoregressive dynamics
        #nb sequences
        S = len(self.initial_values) 
                
        #------vector of means estimations
        self.psi["means"] = self.initial_values[0][:,0]
        for s in range(1, S):
            self.psi["means"] = self.psi["means"] + self.initial_values[s][:,0]
            
        self.psi["means"] = self.psi["means"] / S
              
        #------variance-covariance matrix estimation
        tmp_vec = self.initial_values[0][:,0] - self.psi["means"] 
        tmp_vec = tmp_vec.reshape((self.order,1))
        
        self.psi["covar"] = np.matmul(tmp_vec, np.transpose(tmp_vec))
        for s in range(1, S):
            tmp_vec = self.initial_values[s][:,0] - self.psi["means"]
            tmp_vec = tmp_vec.reshape((self.order,1))
            self.psi["covar"] = self.psi["covar"] + \
                                np.matmul(tmp_vec, np.transpose(tmp_vec))
                                        
        self.psi["covar"] = self.psi["covar"]/S
     
        
    ## @fn
    #  @brief compute log-likelihood of the sequence of initial values
    #
    def init_val_ll(self):
        
        #----------no autoregressive dynamics
        if(self.order == 0):
            return 0.0
        
        #---------- self.order autoregressive dynamics
        #nb sequences
        S = len(self.initial_values) 
        
        #if few sequences have been observed
        if(S < 10):
            return 0.0
        else:

            ll = 0.0
            for s in range(S):
                ll = ll + np.log(multivariate_normal.pdf(\
                        x=self.initial_values[s][:,0] , \
                        mean=self.psi["means"], cov=self.psi["covar"], \
                        allow_singular=False))
                
            return ll
    
    
    ## @fn total_likelihood_s
    #
    #  @param s
    #  
    #  @return LL likelihood matrix of dimension T_s x nb_regime where: \n
    #  * LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ; \theta^{(X,z)}) \n
    #  * \theta^{(X,z)} is the set of parameter related to z^th regime.
    #
    def total_likelihood_s(self, s):
        
        #assertion
        assert (s >= 0 and s < len(self.data))
        
        #final result initialization
        T_s = self.data[s].shape[0]
        LL = np.zeros(dtype=np.float64, shape=(T_s, self.nb_regime))
        
        parameters_k = np.zeros(dtype=np.float64, shape=self.order+2)
        
        for k in range(self.nb_regime):           
            
            parameters_k[0:self.order] = self.coefficients[:, k]
            parameters_k[self.order] = self.sigma[k]                
            parameters_k[self.order+1] = self.intercept[k]
            
            LL[:, k] = likelihood_k(parameters_k, self.data, \
                          self.initial_values, s, self.order, self.innovation)
               
        return LL
                        
                
    ## @fn update_parameters
    #
    #  @brief Computes the step M-X of EM algorithm.
    #
    #  @param list_Gamma A list of matrix where list_Gamma[s] is a T_sxM 
    #  matrix of time dependent marginal a posteriori probabilities relative 
    #  to s^th observed sequence, s in 1, ..., S, with S the number of 
    #  observed sequences.
    #  
    def update_parameters(self, list_Gamma):
        
        if(self.innovation != "gaussian"):
            print()
            print("Error: file regime_switching_ARM.py: given distribution is not supported!")
            exit(1)
        
        #-------------Update regime parameters in parallel  
        #### BEGIN multi-process parallel execution 
        futures = []        
        
        with concurrent.futures.ProcessPoolExecutor(max_workers=self.nb_regime) as executor:               
            #lauch tasks
            for k in range(self.nb_regime):
                futures.append( executor.submit(call_update_parameters_regime_k, \
                                               self, list_Gamma, k) )      
                
            #collect the results as tasks complete
            for f in concurrent.futures.as_completed(futures):
                #update regime k parameters
                (coefficients_k, intercept_k, sigma_k, k) = f.result()                  
                self.intercept[k] = intercept_k
                self.sigma[k] = sigma_k     
                if(self.order != 0):
                    self.coefficients[:,k] = coefficients_k                
        #### End multi-process parallel execution    
                                
        return 
    
    
    ## @fn update_parameters_regime_k
    #  @brief Estimate regime k parameters using a numerical optimatization 
    #   method. In Gaussian innovation, AR coefficients are searched within R: 
    #   if there is a d order unit root then the AR-process is d order integrated.
    #   intercept is within R and sigma is within R+^*
    #   
    #
    #  @param list_Gamma
    #  @param k
    #
    #  @return The estimation of regime k parameters and the index of regime
    #  used in main process to properly collect the results of child processes.
    #   * coefficients_k order length array
    #   * intercerpt_k real number
    #   * sigma_k real number
    #   * k
    #
    def update_parameters_regime_k(self, list_Gamma, k):
        
        #-------parameter bounds: they are searched within [lower_b, upper_b].
        lower_b = np.repeat(-np.inf, self.order+2)
        #standard deviation is positive
        lower_b[self.order] = 1e-5          
        upper_b = np.repeat(np.inf, self.order+2)               
        #parameter bounds
        bounds_ = optimize.Bounds(lb=lower_b, ub=upper_b, keep_feasible=True) 
        
        #-------Q_X_k maximization
        #---Initial guess passed to the optimization algorithm is equal to the 
        #current estimate of parameters
        parameters_init = np.zeros(dtype=np.float64, shape=self.order+2)
        parameters_init[self.order] = self.sigma[k]
        parameters_init[self.order+1] = self.intercept[k]
        if(self.order != 0):
            parameters_init[0:self.order] = self.coefficients[:, k]           
            
        #---Numerical optimization begins
        #NB: default value of eps (step size in gradient approximation) is 1e-08.
        # Notice that this approximation error tends to zero at speed h^2.
        # When convergence fails, this constraint is progressively released 
        # until value 1e-4 o(1e-6) in order to avoid numerical problem.
        # e.g. the failure in gradient numérical calculation (gradient does not
        # match function, abnormal termination of line search LNSRCH).
        epsilon = 1e-08
        conv = False
        while(not conv and epsilon <= 1e-3):
            res = optimize.minimize(fun=Q_X_k, x0=parameters_init, \
                                    args=(k, self.data, self.initial_values, \
                                          self.order, self.innovation, \
                                          list_Gamma),   \
                                    method="L-BFGS-B", jac="2-point", \
                                    bounds=bounds_, \
                                    options={'disp': None, 'maxcor': 10, \
                                             'ftol': 2.220446049250313e-09, \
                                             'gtol': 1e-05, 'eps': epsilon, \
                                             'maxfun': 15000, 'maxiter': 15000, \
                                             'iprint': -1, 'maxls': 20})                                       
            conv = res.success
            epsilon = epsilon * 10
                
        #---WARNING
        if(not res.success):
            print("Warning: numerical optimization: update parameters regime {}, convergence failed with message: {} ".format(k, res.message))
            
        #---return parameters' estimation 
        coefficients_k = []
        if(self.order != 0):
            coefficients_k = res.x[0:self.order]          
    
        sigma_k = res.x[self.order]
        intercept_k = res.x[self.order+1]
               
        return (coefficients_k, intercept_k, sigma_k, k)
                  
       
############################### BEGIN UTILS FUNCTIONS 
        
## @fn call_update_parameters_regime_k
#  @brief Call function update_parameters_regime_k on the given rsvarm object
#
#  @param rsvarm_object
#  @param list_Gamma
#  @param k
#
def call_update_parameters_regime_k(rsvarm_object, list_Gamma, k):
    return rsvarm_object.update_parameters_regime_k(list_Gamma, k) 
    

## @fn pdf
#  For a gamma distribution parameterized by shape parameter a_, and scale
#  parameter scale_, mean = a_ x scale_ and variance = a_ x scale_^2.
#  
#  @param distribution Only "gaussien" distribution is supported
#  @param x Value within the support of the distribution defined by 
#  innovation attribute
#  @param mean Mean of the distribution
#  @param sd Standard deviation of the distribution
#
#  @return If distribution is supported and parameters are valid 
#  returns P(x) ; otherwise returns nan.
#
def pdf(distribution, x, mean, sd):
    
    if (distribution == "gaussian"):
        #if sd <= 0 then norm.pdf returns nan value
        den_x = norm.pdf(x, loc=mean, scale=sd)
    
    else:
        print("*************************************************************************")
        print("Error: file regime_switching_ARM.py: given distribution is not supported!") 
        
        den_x = np.nan
            
    return den_x
            
    
## @fn likelihood_k_order_null
#  @brief Compute the likelihood of observation within regime s when
#   no autoregressive dynamics are included
#
#  @param parameters_k, 2-length vector,parameters_k[0] = standard deviation, 
#   parameters_k[1] = intercept.
#
#  @param data List of length S, where S is the numbers of observation
#  sequences, data[s] is the s^th sequence.
#
#  @param s 
#  @param distribution
#
#  @return Likelihood of data[s] within k^th regime, which is a vector of
#  length T_s.
# 
def likelihood_k_order_null(parameters_k, data, s, distribution):
          
    #assertion
    assert (s >= 0 and s < len(data))

    #final result initialization
    T_s = data[s].shape[0]
    LL = np.ones(dtype=np.float64, shape=T_s) * (-1.0)
            
    for t in range(0, T_s):                
        cond_mean = parameters_k[1]      
        LL[t] = pdf(distribution, data[s][t, 0], cond_mean, parameters_k[0])       
        #assertion
        assert(not math.isnan(LL[t]))
        assert(LL[t] >= 0.)
        
    return LL


## @fn likelihood_k
#
#  @param parameters_k, order+2 length vector, parameters_k[0:order] = autoregressive 
#  coefficients from \phi_{1,k} to \phi_{order,k}, 
#  parameters_k[order] = standard deviation, parameters_k[order+1] = intercept.
#
#  @param data List of length S, where S is the numbers of observation
#  sequences, data[s] is the s^th sequence.
#
#  @param initial_values List of length S, where S is the numbers of observed
#  sequences, initial_values[s] is a matrix orderx1 of initial values associated
#  with s^th sequence.
#
#  @param s 
#  @param order
#  @param distribution
#
#  @return Likelihood of data[s] within k^th regime, which is a vector of
#  length T_s.
# 
def likelihood_k(parameters_k, data, initial_values, s, order, distribution):
  
    #-------------no autoregressive dynamics
    if(order == 0):
        return likelihood_k_order_null(parameters_k, data, s, distribution)

    #-------------autoregressive dynamics
    #assertion
    assert (s >= 0 and s < len(data))

    #final result initialization
    T_s = data[s].shape[0]
    LL = np.ones(dtype=np.float64, shape=T_s) * (-1.0)
    
    #vertical concatenation of data and initial_values
    total_data = np.vstack((initial_values[s], data[s]))
        
    for t in range(order, T_s+order):                
        cond_mean = 0.0
        for i in range(1, order+1):
            cond_mean = cond_mean + parameters_k[i-1] * total_data[t-i, 0]

        cond_mean = cond_mean + parameters_k[order+1]      
        LL[t-order] = pdf(distribution, total_data[t, 0], cond_mean, \
                              parameters_k[order])
        
        #assertion
        assert(not math.isnan(LL[t-order]))
        assert(LL[t-order] >= 0.)
        #continuous PDF can be greater than 1 as an integral within an 
        #interval. Only mass function are bordered within [0, 1].
        
    return LL
   

## @fn Q_X_k
#  @brief
#
#  @param parameters_k 1D array of Order+2 vector with:
#   * parameters[0:order] = autoregressive coefficients
#   * parameters[order] = standard deviation
#   * parameters[order+1] = intercept
#  @param k Regime in which the likelihood is computed.
#  @param data List of length S, where S is the number of observed
#  sequences and data[s] is a s^th sequence.
#  @param initial_values List of length S, where S is the number of observed
#  sequences, initial_values[s] is a matrix orderx1 of initial values associated
#  with s^th sequence.
#  @param order
#  @param distribution
#  @param list_Gamma A list of matrix where list_Gamma[s] is a T_sxM 
#  matrix of time dependent marginal a posteriori probabilities relative 
#  to s^th observed sequence, s in 1, ..., S, with S the number of 
#  observed sequences.
#
#  @return minus log-likelihood of data within k^th regime.
# 
def Q_X_k(parameters_k, k, data, initial_values, order, distribution, list_Gamma):
  
    S = len(data)    
    sum_log_ll = 0.0
    
    for s in range(S):
       
        LL_s = likelihood_k(parameters_k, data, initial_values, s, order, \
                            distribution)            
        #add 1e-308 before taking the log in order to avoid nan value                     
        LL_s = np.log(LL_s + 1e-308) * list_Gamma[s][:, k]
        
        sum_log_ll = sum_log_ll + np.sum(LL_s)
         
    return -sum_log_ll  

############################### END UTILS FUNCTIONS 
