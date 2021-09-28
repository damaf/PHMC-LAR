import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import math
from scipy.stats import norm, gamma, multinomial
#from regime_switching_ARM import likelihood_k
from regime_switching_ARM import pdf

############################################################################
## @package Util functions
# 
#


#/////////////////////////////////////////////////////////////////////////////
# Sample likelihood computing, used in inference algorithm
#/////////////////////////////////////////////////////////////////////////////
## @fn likelihood_k
#
#  @param parameters_k, order+2 vector, parameters_k[0:order] = autoregressive 
#  coefficients from \phi_{1,k} to \phi_{order,k}, 
#  parameters_k[order] = standard deviation, parameters_k[order+1] = intercept
#  @param Obs_seq Tx1 array
#  @param initial_values oderx1 array
#  @param order
#  @param distribution
#
#  @return Likelihood of data[s] within k^th regime, which is a vector of
#  length T_s.
# 
def compute_LL_k(parameters_k, Obs_seq, initial_values, order, distribution):
          
    #final result initialization
    T = Obs_seq.shape[0]
    LL = np.ones(dtype=np.float128, shape=T) * (-1.0)
    
    #vertical concatenation of data and initial_values
    total_data = np.vstack((initial_values, Obs_seq))
        
    for t in range(order, T+order):                
        cond_mean = np.sum(parameters_k[0:order] * \
                                np.flip(total_data[(t-order):t, 0]))
        cond_mean = cond_mean + parameters_k[order+1]      
        LL[t-order] = pdf(distribution, total_data[t, 0], cond_mean, \
                              parameters_k[order])
        
        #assertion
        assert(not math.isnan(LL[t-order]))
        assert(LL[t-order] >= 0)
        #continuous PDF can be greater than 1 as an integral within an 
        #interval. Only mass function are bordered within [0, 1].
        
    return LL


## @fn
#
#  @param coefficients RS-AR process coefficients, matrix AR_order x M
#  @param intercept RS-AR process intercept, M length array 
#  @param sigma RS-AR process standard deviation, M length array 
#  @param innovation Process innovation
#  @param Obs_seq Observation sequence, Tx1 colomn vector
#
#  @return likelihood of Obs_seq, a matrix (T-d)xM, where T equals sequence 
#  length, M the number of regimes and d RS-AR order.
#  LL[t,z] = g(x_t | x_{0}^{t-1}, Z_t, \theta_{z_t})
#  theta_z is the set of parameter related to the z^th regime
#
def compute_LL(coefficients, intercept, sigma, innovation, Obs_seq):
    
    # AR process order and number of regimes
    (order, M) = coefficients.shape
    # Effective number of observations
    T = Obs_seq.shape[0] - order
    assert(T > 0) #observed sequence length is greater than AR order
    
    # initialization
    LL = np.zeros(shape=(T, M), dtype=np.float128) 
    initial_values = Obs_seq[0:order]
    data = Obs_seq[order:]
    
    # k^th regime parameters
    parameters_k = np.zeros(dtype=np.float64, shape=order+2)  
    
    for k in range(M):                
        # k^th regime parameters
        parameters_k[0:order] = coefficients[:, k]
        parameters_k[order] = sigma[k]                
        parameters_k[order+1] = intercept[k]        
        LL[:, k] = compute_LL_k(parameters_k, data, initial_values, \
                                 order, innovation)
   
    return LL
    

#/////////////////////////////////////////////////////////////////////////////
#        ERROR METRICS: INFERENCE
#/////////////////////////////////////////////////////////////////////////////

## @fn 
#  @brief Compute the mean percentage error between two partitions
#
#  @param data_S1 List of N states sequences, each is a 1xT arrays.
#   Reference labels.
#  @param data_S2 List of N states sequences, each is a 1xT arrays.
#   Inferred labels.
#
#  @return N-length array
#
def mean_percentage_error(data_S1, data_S2):
    
    #nb sequences
    N1 = len(data_S1)
    N2 = len(data_S2)  
    assert(N1 == N2)
    
    mean_errors = -1 * np.ones(shape=N1, dtype=np.float64)
    
    for s in range(N1):
        
        #sequence length
        T1 = data_S1[s].shape[1]
        T2 = data_S2[s].shape[1]  
        assert(T1 == T2)        
    
        assert(np.sum(data_S1[s][0,:] < 0) == 0)
        assert(np.sum(data_S2[s][0,:] < 0) == 0)
                
        mean_errors[s] =  np.mean(data_S1[s][0,:] != data_S2[s][0,:])
               
    return mean_errors


#/////////////////////////////////////////////////////////////////////////////
#        ERROR METRICS: FORECASTING
#/////////////////////////////////////////////////////////////////////////////

## @fn
#  @brief Computes 
#   Mean Absolute Percentage Error (MAPE)
#   Root Mean Square Error (RMSE)
#   Biais metric 
#
#  @param X_obs List of N Hx1 arrays.
#  @param X_pred List of N Hx1 arrays.
#
#  @return Hx3 array
#
def compute_forecast_error(X_obs, X_pred, H):
    
    #nb sequences
    N = len(X_obs)
    assert(N == len(X_pred))
    
    assert(N == 1) 
    
    mean_errors = np.zeros(shape=(H,3), dtype=np.float64)
    
    for h in range(H):
               
        tmp_mae = 0.0
        tmp_rmse = 0.0
        tmp_bias = 0.0
        
        for s in range(N):
            #MAPE
            tmp_mae = tmp_mae + np.abs( (X_obs[s][h,0] - X_pred[s][h,0]) / X_obs[s][h,0] )
            #RMSE
            tmp_rmse = tmp_rmse + (X_obs[s][h,0] - X_pred[s][h,0])**2
            #Bias
            tmp_bias = tmp_bias + (X_obs[s][h,0] - X_pred[s][h,0])
                
            #print(X_obs[s][h,0] - X_pred[s][h,0])
                
        mean_errors[h, 0] = tmp_mae/N
        mean_errors[h, 1] = np.sqrt(tmp_rmse/N)
        mean_errors[h, 2] = tmp_bias/N
                  
    return mean_errors     










       
