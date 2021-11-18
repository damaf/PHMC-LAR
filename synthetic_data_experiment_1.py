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
#   * P The percentage of labelled data.
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
#  @param data_S List of N states sequences, each is a 1xT arrays.
#  @param nb_regimes The number of classes with data.
#  @param P The percentage of labelled data.
#
#  @return List of N of 1xT arrays
#
def create_partially_labelled_seq(data_S, nb_regimes, P):
          
    #nb sequence
    N = len(data_S)
    #sequence length
    T = data_S[0].shape[1]   
    
    #---supervised learning
    if(P == 100):
        res = []
        for s in range(N):
            res.append( -1*np.ones(dtype=np.int32, shape=(1,T)) )
            for t in range(T):
                res[s][0, t] = data_S[s][0,t]
            
        return res
    
    #---unsupervised learning
    elif(P == 0):
        res = []
        for _ in range(N):
            res.append( -1*np.ones(dtype=np.int32, shape=(1,T)) )
            
        return res 
    
    #---semi-supervised learning
    else:
        res = []
        nb_labelled = int(np.round(P*T/ 100))
        
        for s in range(N):
        
            res.append( -1*np.ones(dtype=np.int32, shape=(1,T)) )       
            labelled_ind = np.random.choice([i for i in range(T)], \
                                             size=nb_labelled, replace=False)
                
            for t in labelled_ind:            
            
                res[s][0, t] = data_S[s][0, t]  
              
    return res


#=====================================Begin script
#starting time 
start_time = time.time()


#----three command line argument are required
if(len(sys.argv) != 4):
    print("ERROR: script unreliable_labels.py takes 4 arguments !")
    print("Usage: ./synthetic_data_experiment_1.py train_data_file output_dir P")
    sys.exit(1)


#----percentage of labelled observations
#percentage of labelled data
P = float(sys.argv[3])

if(P > 100 or P < 0):
    print("ERROR: script synthetic_data_experiment_1.py: 0 <= P <= 100 !")
    sys.exit(1)
    

#----hyper-parameters setting
X_order = 2
nb_regimes = 4
innovation = "gaussian"


#----training data loading
infile = open(sys.argv[1],'rb')
data_set = pickle.load(infile)
infile.close()

#time series and associated initial values
data_X = data_set[1]
initial_values = data_set[2]


#create unreliable labels
data_S = create_partially_labelled_seq(data_set[0], nb_regimes, P)


#----LOG-INFO
print("**********************************************************************")
print("S = {}, T = {}".format(len(data_S), data_S[0].shape[1]))
print("X_order = {}, nb_regimes = {}, innovation = {}".format(X_order,  \
       nb_regimes, innovation))
print("train_data_file = {}, output_dir = {}".format(sys.argv[1], sys.argv[2]))
output_file = os.path.join(sys.argv[2], os.path.basename(sys.argv[1])) + \
                "_reliable-label_P=" + str(P)
print("Trained model has been saved within file {}".format(output_file))

print("********************Partial labelling*********************************")
N = len(data_set[0])
state_0 = 0
state_1 = 0
state_2 = 0
state_3 = 0
for s in range(N):
    state_0 = state_0 + np.sum(data_S[s] == 0)
    state_1 = state_1 + np.sum(data_S[s] == 1)
    state_2 = state_2 + np.sum(data_S[s] == 2)
    state_3 = state_3 + np.sum(data_S[s] == 3)
print("P = {}, state_0 = {}, state_1 = {}, state_2 = {}, state_3 = {}, total labelled = {}".format(\
      sys.argv[3], state_0, state_1, state_2, state_3, (state_0+state_1+state_2+state_3) ))
    
    
#----learning
model_output = hmc_lar_parameter_learning (X_order, nb_regimes, data_X, \
               initial_values, data_S, innovation)


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











