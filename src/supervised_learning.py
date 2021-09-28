import numpy as np
from gaussian_innovation import Gaussian_X


###############################################################################
# Here we consider the supervised learning case in which states are observed.
# This corresponds to Observed Regime Swithwing AR model.
###############################################################################

## @fn trans_freq_s
#  @brief Compute the frequency of each states within the given state sequence.
#
#  @param states State sequence, 1xT array.
#  @param M number of regimes.
#  @param T Sequence length.
#
#  @return MxM array.
#
def trans_freq_s(seq_S, M, T):
    
    mat = np.zeros(dtype=np.float64, shape=(M, M))
    
    for t in range(1, T):
        mat[seq_S[0, t-1], seq_S[0, t]] += 1   

    return mat
   
    
## @fn compute_trans_freq
#  @brief Compute the frequency of each states using the given list of states 
#   sequences. 
#
#  @param states List of N 1xT arrays of state sequences.
#  @param M number of regimes.
#
#  @return MxM array
#
def compute_trans_freq(states, M):
    
    N = len(states)
    T = states[0].shape[1]
    F = np.zeros(dtype=np.float64, shape=(M, M))

    for s in range(N):
        F = F + trans_freq_s(states[s], M, T)
        
    return F


## @fn compute_gamma
#  @brief This function computes the smoothed marginal probabilities from the 
#   observed states as bellow:
#   * Gamma[t,i] = P(Z_t=z | X, Sigma, Theta) = 1 if Z_t = z;
#   * 0 otherwise.
#
#  @param states List of N 1xT arrays of state sequences.
#  @param M number of regimes.
#
#  @return List of N TxM arrays.
#
def compute_gamma(states, M):
    
    list_Gamma = []
    N = len(states)
    T = states[0].shape[1]
    
    for s in range(N):
        list_Gamma.append( np.zeros(dtype=np.float64, shape=(T, M)) )
        
        for t in range(T):
            list_Gamma[s][t, states[s][0, t]] = 1
     
    return list_Gamma
    

## @fn compute_HMC_parameter
#  @brief 
#
#  @param F MxM matrix of transition frequency.
#  @param list_Gamme
#
#  @return (A, Pi)
#    * A MxM transition matrix.
#    * Pi 1xM initial probabilities.
#
def compute_HMC_parameter(F, list_Gamma):
    
    S = len(list_Gamma)
    (T, M) = list_Gamma[0].shape
    
    A = np.zeros(dtype=np.float64, shape=(M, M))
    Pi = np.zeros(dtype=np.float64, shape=(1, M))
    
    #---------------Pi
    for i in range(M):    
        
        aux = 0.0
        for s in range(S):
            aux = aux + list_Gamma[s][0,i]
            
        Pi[0, i] = aux / S
        
    Pi[0, M-1] = 1.0 - np.sum(Pi[0, 0:(M-1)])
        
    #---------------A
    for i in range(M):
        
        state_i_freq = 0.0
        for s in range(S):
            state_i_freq = state_i_freq + np.sum(list_Gamma[s][:, i])
            
        A[i, :] = F[i, :] / state_i_freq
        A[i, M-1] = 1.0 - np.sum(A[i, 0:(M-1)])
        
        
    return (A, Pi)
    

## @fn
#  @brief Computes the parameters of Observed Regime Swithwing AR model.
#
#  @param X_order Autoregressive order, must be positive.
#  @param nb_regimes Number of switching regimes.
#
#  @param data List of length S, where S is the number of observed
#  time series. data[s], the s^th time series, is a column vector
#  T_sx1 where T_s denotes its size starting at timestep t = order + 1 included.
#
#  @param initial_values List of length S. initial_values[s] is a column vector 
#   orderx1 of initial values associated with the s^th sequence.
#
#  @param states List of S fully observed state sequences. 
#   states[s] is a line vector 1xT taking values in {0, 1, 2, ..., M-1} where
#   M is the number of regimes.
#
#  @param innovation Law of model error terms, only 'gaussian' noises are supported  
#
#  @return The maximum likelihood estimate of parameters.
#
def learning(X_order, nb_regimes, data, initial_values, states, innovation):
    
    #---------------AR process initialization  
    if (innovation == "gaussian"):
        X_process = Gaussian_X(X_order, nb_regimes, data, initial_values, \
                               states)              
    else:
        print()
        print("ERROR: file EM_learning.py: the given distribution is not supported!")
        
    #---------------Learning
    #---learn HMC parameter
    F = compute_trans_freq(states, nb_regimes)
    list_Gamma = compute_gamma(states, nb_regimes)
    (A, Pi) = compute_HMC_parameter(F, list_Gamma)
    
    #---learn AR process parameters
    X_process.estimate_psi_MLE()
    X_process.update_parameters(list_Gamma)
    
    #---------------compute log_ll
    S = len(states)
    T = states[0].shape[1]
    
    log_ll = 0.0
    for s in range(S):
        
        LL_s = X_process.total_likelihood_s(s)
        for t in range(T):
            log_ll = log_ll + np.log( np.sum(LL_s[t, :] * list_Gamma[s][t, :]) )

    log_ll = log_ll + X_process.init_val_ll()
    
    return (log_ll, A, Pi, list_Gamma, [], X_process.coefficients, \
            X_process.sigma, X_process.intercept, X_process.psi)
    
    
    
