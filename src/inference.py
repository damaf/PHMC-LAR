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
      

