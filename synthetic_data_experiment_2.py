#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 22:55:36 2020

@author: dama-f
"""

##############################################################################
#  This experiment is to learn PHMC-LAR model on unreliable labbeled data.
#  Unreliablility level is controlled by parameter rho.
#  rho is the mean of a Beta law that draws the probability for a label to be
#  wrong.
#  This is an supervised learning problem.
#
#  This script take three input parameters
#   * train_data_file This is a pickle serialization file that contains the
#     training data.
#   * output_dir The name of the directory in which the trained model has to be 
#     saved.
#   * rho The level of unreliability upon labels.
#
##############################################################################

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), \
                            "src")))
import time
import pickle
import numpy as np
from scipy.stats import beta, bernoulli

from supervised_learning import learning

import warnings
warnings.filterwarnings("ignore")


## @fn
#  @brief Sample non-reliable states from the given data set of N sequences.
#   For each sequence, the true states are replaced by wrongs ones with 
#   probabilities drawn from a Beta(rho, sigma) law. Each wrong state is
#   uniformly chosen from the other states execpt the true one.
#
#  @param data_S List of N states sequences, each is a 1xT arrays.
#  @param nb_regimes The number of classes with data.
#  @param rho Unreliability level expressed in percentage. 
#   This is the mean of Beta distribution.
#  @param sigma Unreliability level standard deviation.
#   This is the standard deviation of Beta distribution.   
#
#  @warning: error occurrs when rho > 95
#
#  @return List of N of 1xT arrays
#
def create_unreliable_labels(data_S, nb_regimes, rho, sigma):
       
    #----no labelling error
    if(rho == 0):
        return data_S
      
    #----labelling error
    #nb sequence
    N = len(data_S)
    #sequence length
    T = data_S[0].shape[1]    
    #output 
    res = []
    
    #parameters of Beta distribution that mean and variance equal rho and
    #sigma^2
    rho = rho / 100.0
    a = rho * (rho - rho**2 - sigma**2) / sigma**2
    b = (1 - rho) * (rho - rho**2 - sigma**2) / sigma**2
    
    for i in range(N):
        
        res.append((-2)*np.ones(dtype=np.int32, shape=(1,T)))
        
        for t in range(T):            
            #labelling error
            error_prob = beta.rvs(a, b)
            #this is equals 1 if labelling error occurs, 0 otherwise
            error = bernoulli.rvs(error_prob)
            
            if(error == 1):
                wrong_states = [ s for s in range(nb_regimes) \
                                                    if s != data_S[i][0, t] ]
                #true label is replaced by a random label 
                res[i][0, t] = np.random.choice(wrong_states)
                
            elif(error == 0):
                res[i][0, t] = data_S[i][0, t]
                
            else:
                print("ERROR: create unreliable labels, error = {}".format(error))
                sys.exit(1)
                         
    return res


#=====================================Begin script
#starting time 
start_time = time.time()


#----three command line argument are required
if(len(sys.argv) != 4):
    print("ERROR: script unreliable_labels.py takes 3 arguments !")
    print("Usage: ./synthetic_data_experiment_2.py train_data_file output_dir rho")
    sys.exit(1)


#----unreliability level of labels, given in percentage
#mean of beta law, parameter given in command line
rho = float(sys.argv[3])

if(rho >= 100 or rho < 0):
    print("ERROR: script synthetic_data_experiment_2.py: 0 <= rho < 100 !")
    sys.exit(1)
    
#standard deviation of beta law, fixed. 
#This value has been used by Ramesso and Denoeux
sigma = 0.2


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
data_S = create_unreliable_labels(data_set[0], nb_regimes, rho, sigma)


#----LOG-INFO
print("**********************************************************************")
print("S = {}, T = {}".format(len(data_S), data_S[0].shape[1]))
print("X_order = {}, nb_regimes = {}, innovation = {}".format(X_order,  \
       nb_regimes, innovation))
print("train_data_file = {}, output_dir = {}, rho = {}".format(sys.argv[1], \
       sys.argv[2], sys.argv[3]))

output_file = os.path.join(sys.argv[2], os.path.basename(sys.argv[1])) + \
                "_unreliable-label_rho=" + str(rho)
print("Trained model has been saved within file {}".format(output_file))
#
print("***********************Labelling errors*******************************")
T = data_S[0].shape[1]
S = len(data_S)
error_rate = 0
for s in range(S):
    error_rate = error_rate + np.sum(data_set[0][s][0,:] != data_S[s][0,:])
print("Labelling error rate = {}".format(error_rate / (T*S)))


#----learning: without initialization procedure
"""
model_output = learning (X_order, nb_regimes, data_X, initial_values, \
                         data_S, innovation)
"""

#eviquently, we can use EM_learning which will converge in one iteration.
from EM_learning import hmc_lar_parameter_learning
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
