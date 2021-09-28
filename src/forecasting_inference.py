#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 19:11:11 2020

@author: dama-f
"""

import numpy as np
import pickle
from utils import compute_LL


#/////////////////////////////////////////////////////////////////////////////
#           INFERENCE 
#/////////////////////////////////////////////////////////////////////////////
    
## @fun viterbi
#  Maximum A Posterio classification: P(Z,X) = P(Z|X)*P(X)
#   
#  @param A
#  @param Pi
#  @param coefficients
#  @param intercepts 
#  @param sigma
#  @param innovation Error term's law
#  @param Obs_seq Sequence to be labelled, Tx1 colomn vector
#  @param Obs_states 1xT colomn vector T, with Sigma[0,t] = -1 if Z_t is 
#   latent otherwise. Sigma[t,0] in {0, 1, 2, ..., M-1}.
#  
#  @return 1xT array
#
def viterbi(A, Pi, coefficients, intercepts, sigma, innovation, Obs_seq, \
            Obs_states):
    
    #number of states
    M = A.shape[0]     
    #compute LL, TxM matrix
    LL = compute_LL(coefficients, intercepts, sigma, innovation, Obs_seq)
    #effective length of observation sequence
    T = LL.shape[0]
           
    #------no observed states
    if(Obs_states == []):
        Obs_states = -1 * np.ones(shape=(1, T), dtype=np.int32)
        
    #------initialize probability matrix D
    D = np.zeros(shape=(T,M), dtype=np.float64)
    
    #------initial D probabilities
    if(Obs_states[0, 0] == -1): #no observed states at time 1     
        D[0, :] = Pi[0, :] * LL[0, :]
    else:     #state Obs_states[0, 0] has been observed at time 1
        class_1 = Obs_states[0, 0]
        D[0, class_1] = Pi[0, class_1] * LL[0, class_1]
        #all other probabilities D[0, i] are null
           
    #------compute D for t=1,...,T-1
    for t in range(1, T):       
        if(Obs_states[0, t] == -1): #no observed state at time t
            for i in range(M):
                temp_product = A[:, i] * D[t-1, :]
                D[t, i] = np.max(temp_product) * LL[t, i]
        else:   #state Obs_states[0, t] has been observed at time 1
            class_t = Obs_states[0, t]
            temp_product = A[:, class_t] * D[t-1, :]
            D[t, class_t] = np.max(temp_product) * LL[t, class_t]
            #all other probabilities D[t, u] are null
    
    #assertion        
    assert(np.sum(D < 0) == 0)

    #------optimal state computing: backtracking
    opt_states = -1 * np.ones(shape=(1,T), dtype=np.int32)
    opt_states[0, T-1] = np.argmax(D[T-1, :])
    
    for t in range(T-2, -1, -1):
        opt_states[0, t] = np.argmax( D[t, :] * A[:, opt_states[0, t+1]] )
     
        
    return opt_states


## @fun viterbi_log: to use when t > 30
#  @param A
#  @param Pi
#  @param coefficients
#  @param intercepts 
#  @param sigma
#  @param innovation Error term's law
#  @param Obs_seq Sequence to be labelled, Tx1 colomn vector
#  @param Obs_states 1xT colomn vector T, with Sigma[0,t] = -1 if Z_t is 
#   latent otherwise. Sigma[t,0] in {0, 1, 2, ..., M-1}.
#  
#  @return 1xT array
#
def viterbi_log(A, Pi, coefficients, intercepts, sigma, innovation, Obs_seq, \
                Obs_states):
        
    #number of states
    M = A.shape[0]     
    #compute LL, TxM matrix
    LL = compute_LL(coefficients, intercepts, sigma, innovation, Obs_seq)
    #effective length of observation sequence
    T = LL.shape[0]
        
    #------no observed states
    if(Obs_states == []):
        Obs_states = -1 * np.ones(shape=(1,T), dtype=np.int32)
    
    #------compute log probabilities
    tiny = np.finfo(0.).tiny
    A_log = np.log(A + tiny)
    Pi_log = np.log(Pi + tiny)
    LL_log = np.log(LL + tiny)
    
    #------initialize log probability matrix D_log
    D_log = -1e300*np.ones(shape=(T,M), dtype=np.float128)
    
    #------initial D probabilities
    if(Obs_states[0, 0] == -1): #no observed states at time 1     
        D_log[0, :] = Pi_log[0, :] + LL_log[0, :]
    else:     #state Obs_states[0, 0] has been observed at time 1
        class_1 = Obs_states[0, 0]
        D_log[0, class_1] = Pi_log[0, class_1] + LL_log[0, class_1]
        #all other probabilities D_log[0, i] is -inf
    
    #------compute D for t=1,...,T-1
    for t in range(1, T):       
        if(Obs_states[0, t] == -1): #no observed state at time t
            for i in range(M):
                temp_sum = A_log[:, i] + D_log[t-1, :]
                D_log[t, i] = np.max(temp_sum) + LL_log[t, i]
        else:   #state Obs_states[0, t] has been observed at time 1
            class_t = Obs_states[0, t]
            temp_sum = A_log[:, class_t] + D_log[t-1, :]
            D_log[t, class_t] = np.max(temp_sum) + LL_log[t, class_t]
            #all other probabilities D_log[t, u] are -inf
    
    #assertion        
    assert(np.sum(D_log < -1e300) == 0)
    
    #------optimal state computing: backtracking
    opt_states = -1 * np.ones(shape=(1,T), dtype=np.int32)
    opt_states[0, T-1] = np.argmax(D_log[T-1, :])
    
    for t in range(T-2, -1, -1):
        opt_states[0, t] = np.argmax( D_log[t, :] + A_log[:, opt_states[0, t+1]] )
     
        
    return opt_states


## @fn 
#  @brief
#
#  @param model_file
#  @param innovation
#  @param list_Obs_seq
#  @param list_Obs_states
#  @param log_version
#
#  @return
#
def inference(model_file, innovation, list_Obs_seq, list_Obs_states=[], \
              log_version=True):
    
    #-----model loading
    infile = open(model_file, 'rb')
    phmc_lar = pickle.load(infile)
    infile.close()
    
    #-----required phmc_lar parameters
    A = phmc_lar[1]
    Pi = phmc_lar[2]
    ar_coefficients = phmc_lar[5]
    ar_intercepts = phmc_lar[7]  
    Sigma = phmc_lar[6]   
        
    #assertion
    assert(np.sum(A < 0.0) == 0)
    assert(np.sum(Pi < 0.0) == 0)
    
    #nb sequence
    N = len(list_Obs_seq)
    
    #-----no observed state
    if(len(list_Obs_states) == 0):
        list_Obs_states = [ [] for _ in range(N) ] 
    else:
        assert(N == len(list_Obs_states))
        
    #output
    list_states = []
    
    #-----Inference begins
    for s in range(N):
    
        if(log_version):
            list_states.append( viterbi_log(A, Pi, ar_coefficients, \
                                            ar_intercepts, Sigma, \
                                            innovation, list_Obs_seq[s], \
                                            list_Obs_states[s]) )
        else:
            list_states.append( viterbi(A, Pi, ar_coefficients, \
                                        ar_intercepts, Sigma, \
                                        innovation, list_Obs_seq[s], \
                                        list_Obs_states[s]) )
    
    return list_states
      

#/////////////////////////////////////////////////////////////////////////////
#           FORECASTING
#/////////////////////////////////////////////////////////////////////////////

## @fn compute_means
#  @brief Compute the mean X_{T+h}'s probability law  with each state.
#
#  @param ar_coefficients orderxM array of autoregressive coefficients
#  @param ar_intercepts M length array 
#  @previous_vals order length array of previous values with 
#   previous_vals[j] = X_{T+h-j}
#
#  @return M-length array
#
def compute_means(ar_coefficients, ar_intercepts, previous_vals):
    
    #nb regimes
    M = ar_coefficients.shape[1]
    
    #output
    means = np.zeros(shape=M, dtype=np.float64)
    
    for i in range(M):        
        means[i] = np.sum(ar_coefficients[:, i] * previous_vals) + \
                    ar_intercepts[i]
        
    return means

## @fn compute_prediction_probs
#  @brief Compute probabilities P(Z_{T+h}=k | X_{1-p}^T)
#
#  @param A MxM transition matrix
#  @param previous_pred_prob M-length array of prediction probabilities
#   P(Z_{T+h-1}=k | X_{1}^T)
#  @parameter obs_state The observed state at time-step T+h, equals
#   -1 if Z_{T+h} is latent
#
#  @return M-length array
#
def compute_prediction_probs(A, previous_pred_prob, obs_state):
    
    #nb regimes
    M = A.shape[0]
    
    #output
    curr_pred_probs = np.zeros(shape=M, dtype=np.float64)
        
    #all regimes are possible
    if(obs_state == -1):
        for j in range(M):
            curr_pred_probs[j] = np.sum(A[:, j] * previous_pred_prob)
   
    #obs_state has been observed
    else:
        curr_pred_probs[obs_state] = 1
        #all other probabilities curr_pred_probs[j] are null
        
    #normalization
    curr_pred_probs = curr_pred_probs / np.sum(curr_pred_probs)
        
    assert(np.sum(curr_pred_probs < 0.) == 0)
    assert(np.sum(curr_pred_probs > 1.) == 0)
    
         
    return curr_pred_probs


## @forecasting_one_seq
#  @brief Compute H-step ahead forecasting on the given sequence. 
#   At each time-step T+h, the expectation of X_t knowing X_{1-order}^{t-1} 
#   is computed.
#   
#  @param A
#  @param ar_coefficients
#  @param ar_intercepts
#  @param innovation
#  @param Gamma
#  @param init_values orderX1-array of initial values
#  @param H
#  @param states 1xH array of observed states at forecast horizons
#
#  @return A Hx1-array of predicted values
# 
def forecasting_one_seq(A, ar_coefficients, ar_intercepts, innovation, \
                        Gamma, init_values, H, states):
           
    #if numerical issues arise during learning then Gamma[-1, i] = 0 for all i
    assert(np.sum(Gamma[-1, :]) != 0.0)
    
    #----no observed states at forecast horizons
    if(states == []):
        states = -1 * np.ones(shape=(1, H), dtype=np.int32)
        
    #AR order
    order = ar_coefficients.shape[0]
    
    #output
    predictions = np.zeros(shape=(H, 1), dtype=np.float64)
    
    #probabilities P(Z_T=k | X_1^T) normalization
    previous_pred_prob = Gamma[-1, :]  / np.sum(Gamma[-1, :]) 
 
    #previous value
    previous_vals = np.zeros(shape=order+H, dtype=np.float64)
    assert(init_values.shape[0] == order)
    previous_vals[0:order] = init_values[:, 0]
    
    for h in range(H):    
        
        #---predition probabilities
        curr_pred_probs = compute_prediction_probs(A, previous_pred_prob, \
                                                   states[0,h])
             
        #---prediction
        means = compute_means(ar_coefficients, ar_intercepts, \
                              np.flip(previous_vals[h:(order+h)]))
                                 
        predictions[h, 0] = np.sum(means * curr_pred_probs)
                   
        #---current prediction is added to previous values
        previous_vals[order+h] = predictions[h, 0] 
        
        #---update previous_pred_prob              
        previous_pred_prob = curr_pred_probs
        
                
    return predictions


## @forecasting 
#  @brief Compute H-step ahead forecasting on each sequence of the given list
#   
#  @param model_file Pickle file in which PHMC-LAR model is saved
#  @param list_init_values List of N orderX1-array of initial values where d is AR 
#   process order.
#  @param innovation
#  @param H Forecast horizon 
#  @param list_states list ol N 1xH array, possible states at forecast horizon
#  
#  @return A list of N Hx1-array of predicted values
#   
def forecasting(model_file, list_init_values, innovation, H, list_states=[]):
            
    #model loading
    infile = open(model_file, 'rb')
    phmc_lar = pickle.load(infile)
    infile.close()
    
    # required phmc_lar parameters
    A = phmc_lar[1]
    ar_coefficients = phmc_lar[5]
    ar_intercepts = phmc_lar[7]  
    list_Gamma = phmc_lar[3]   
    
    #assertion
    assert(np.sum(A < 0.0) == 0)
            
    #nb sequences
    N = len(list_init_values)
    
    if(len(list_states) == 0):
        list_states = [ [] for _ in range(N) ] 
    else:
        assert(N == len(list_states))
    
    #output
    list_predictions = []
    
    #for each sequence
    for s in range(N):
        
        list_predictions.append(forecasting_one_seq(A, ar_coefficients, \
                                                    ar_intercepts, \
                                                    innovation, list_Gamma[s], \
                                                    list_init_values[s], H,
                                                    list_states[s])) 
                                                     
    return list_predictions


