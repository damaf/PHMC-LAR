#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:55:36 2020

@author: dama-f
"""

##############################################################################
#  This experiment consists of learning PHMC-LAR model on partially labelled 
#  observations. This is a semi-supervised learning problem.
#
#  This script take three input parameters
#   * train_data_file This is a pickle serialization file that contains the
#     training data.
#   * output_dir The name of the directory in which the trained model has to be 
#     saved.
#   * P Specifies weither partial knowledge have to be included: 0 for unsupervised,
#     1 for partial knowledge, 2 for supervised
#   * D Autoregressive order
#
##############################################################################

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                            "src")))
import time
import pickle
import numpy as np

from EM_learning import hmc_lar_parameter_learning

import warnings
warnings.filterwarnings("ignore")


## @fn
#  @brief Randomly labelled given list of HMC sequences.
#
#  @param data_S List of N states sequences, each is a 1xT_i arrays.
#  @param nb_regimes The number of classes with data.
#  @param P Specifies weither partial knowledge have to be included, 
#   0 for false, 1 for true.
#
#  @return List of N of 1xT arrays
#
def create_partially_labelled_seq(data_S, nb_regimes, P):
          
    #nb sequences
    N = len(data_S)
    
    assert(P == 0 or P == 1)
    assert(nb_regimes == 3)
    
    #---add reliable partial knowledge 
    if(P == 1):
        res = []
        for s in range(N):
            #sequence length
            T_s = data_S[s].shape[1]   
            
            #index of break point 1 and 2
            b1 = np.where(data_S[s][0,:] == 1)[0][0]
            b2 = np.where(data_S[s][0,:] == 2)[0][0]
            
            #size of the interval before and after switching times
            inter_size = 15
            
            res.append( -1*np.ones(dtype=np.int32, shape=(1,T_s)) )
            
            res[s][0, 0:(b1-inter_size)] = data_S[s][0, 0:(b1-inter_size)] 
            res[s][0, (b1+inter_size+1):(b2-inter_size)] = data_S[s][0, (b1+inter_size+1):(b2-inter_size)] 
            res[s][0, (b2+inter_size+1):] = data_S[s][0, (b2+inter_size+1):] 
            
        return res
    
    #---no partial knowledge = unsupervised
    elif(P == 0):
        res = []
        for s in range(N):
            #sequence length
            T_s = data_S[s].shape[1]   
            res.append( -1*np.ones(dtype=np.int32, shape=(1,T_s)) )
            
        return res

    #---fully supervised
    elif(P == 2):
        res = []
        for s in range(N):
            T_s = data_S[s].shape[1] 
            res.append( -1*np.ones(dtype=np.int32, shape=(1,T_s)) )
            for t in range(T_s):
                res[s][0,t] = data_S[s][0,t]

        return res
    
    #---error
    else:    
        print("ERROR: in function create_partially_labelled_seq! \n")
        return {}
    
#=====================================Begin script
#starting time 
start_time = time.time()


#----four command line arguments are required
if(len(sys.argv) != 5):
    print("ERROR: script run_training.py takes 4 arguments !")
    print("Usage: ./cmapss_experiment.py train_data_file output_dir P D")
    sys.exit(1)


#----percentage of labelled observations
#percentage of labelled data
P = int(sys.argv[3])

if(P != 0 and P != 1 and P != 2):
    print("ERROR: argument P takes three values, 0 for unsupervised case, ", \
          "1 for semi-supervised case and 2 for supervised case!")
    sys.exit(1)
    

#----hyper-parameters setting
X_order = int(sys.argv[4])
nb_regimes = 3
innovation = "gaussian"


#----training data loading
infile = open(sys.argv[1],'rb')
data_set = pickle.load(infile)
infile.close()

#model is trained on the first 60 % of trajectoriess
total_trajec = len(data_set[0])
S =  int( np.round(total_trajec*60/100) ) 

#time series, associated initial values and states
data_S_true = [] #list of 1xT_i sequences
data_X = [] #list of T_ix1 sequences
initial_values = [] #list of X_orderx1 sequences
min_T_i = 1e10
max_T_i = -1

for i in range(S):
    
    T_i = data_set[1][i].shape[0] - X_order
    initial_values.append(data_set[1][i][0:X_order,])
    data_X.append(data_set[1][i][X_order:, ])    
    
    tmp = -1*np.ones(dtype=np.int32, shape=(1,T_i))
    tmp[0, :] = data_set[0][i][0, X_order:]
    data_S_true.append( tmp )
    
    if(min_T_i > T_i):
        min_T_i = T_i
    if(max_T_i < T_i):
        max_T_i = T_i

#create partial knowledge
data_S = create_partially_labelled_seq(data_S_true, nb_regimes, P)


#----LOG-INFO
print("**********************************************************************")
print("S = {}, T = {}".format(len(data_S), data_S[0].shape[1]))
print("X_order = {}, nb_regimes = {}, innovation = {}".format(X_order,  \
       nb_regimes, innovation))
print("train_data_file = {}, output_dir = {}".format(sys.argv[1], sys.argv[2]))
output_file = os.path.join(sys.argv[2], "partial-label") + \
                "_P=" + str(P) + "_D=" + str(X_order)
print("Trained model has been saved within file {}".format(output_file))

print("********************Partial labelling*********************************")
state_0 = 0
state_1 = 0
state_2 = 0
for s in range(S):
    state_0 = state_0 + np.sum(data_S[s] == 0)
    state_1 = state_1 + np.sum(data_S[s] == 1)
    state_2 = state_2 + np.sum(data_S[s] == 2)
print("P = {}, D = {}, state_0 = {}, state_1 = {}, state_2 = {}, total labelled = {}".format(\
       P, X_order, state_0, state_1, state_2, (state_0+state_1+state_2) ))
    
    
#----learning
"""
model_output = hmc_lar_parameter_learning (X_order, nb_regimes, data_X, \
               initial_values, data_S, innovation,)
"""
model_output = hmc_lar_parameter_learning (X_order, nb_regimes, data_X, \
               initial_values, data_S, innovation, \
               nb_iters=1, epsilon=1e-6, \
               nb_init=1, nb_iters_init=1)


#----save model_output 
outfile = open(output_file, 'wb') 
pickle.dump(model_output, outfile)
outfile.close()


#----running time estimation ends
duration = time.time() - start_time
print("======================================================================")
print("#learningTime: algorithm lastes {} minutes".format(duration/60))
print("======================================================================")


#----learnt parameters
print("---------------------psi-------------------------")
print(model_output[-1])
print("------------------AR process----------------------")
print("#total_log_ll= ", model_output[0])
print("ar_coefficients=", model_output[5])
print("intercept=", model_output[7])
print("sigma=", model_output[6])
print()
print("------------------Markov chain----------------------")
print("Pi=", model_output[2])
print("A=", model_output[1])










