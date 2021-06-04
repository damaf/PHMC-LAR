#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 13:31:20 2020

@author: dama-f
"""

import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname("./"), "src")))

import numpy as np
import pickle
from sklearn.metrics import confusion_matrix

from inference import inference
from utils import mean_percentage_error


###############################################################################
#     CMAPSS inference task and confusion matrices
###############################################################################


## for training datasets begin_index=0 and nb_seq=60
#  for validation datasets begin_index=60 and nb_seq=20
#  for test datasets begin_index=80 and nb_seq=20
#
def cmapss_inference_task(model_file, dataset_num, begin_index, nb_seq, D):

    X_order = D
    
    #---data loading
    data_file = "data/cmapss-data/train_FD00" + str(dataset_num) + "_HI_state"
    infile = open(data_file,'rb')
    data_set = pickle.load(infile)
    infile.close()

    #---data preprocessing
    data_S = [] #list of 1xT sequences
    total_data_X = [] #list of Tx1 sequences

    for i in range(begin_index, begin_index+nb_seq):
    
        #initial values + sequence to be segmented
        total_data_X.append(data_set[1][i]) 
        
        #true segmentation - except states of initial values
        T_i = data_set[1][i].shape[0] - X_order   
        tmp = -1*np.ones(dtype=np.int32, shape=(1,T_i))
        tmp[0, :] = data_set[0][i][0, X_order:]
        data_S.append( tmp )
    
    #---inference
    inf_states = inference(model_file, "gaussian", total_data_X)
        
    errors = mean_percentage_error(data_S, inf_states)
    
    return (inf_states, errors)



def confusion(model_file, dataset_num, D, state_number=[]):
    X_order = D
    
    #---data loading
    data_file = "data/cmapss-data/train_FD00" + str(dataset_num) + "_HI_state"
    infile = open(data_file,'rb')
    data_set = pickle.load(infile)
    infile.close()
    
    #---data preprocessing
    data_S = [] #list of 1xT sequences
    total_data_X = [] #list of Tx1 sequences
    
    N = 20
    for i in range(80, 80+N):
    
        #initial values + sequence to be segmented
        total_data_X.append(data_set[1][i]) 
        
        #true segmentation - except states of initial values
        T_i = data_set[1][i].shape[0] - X_order   
        tmp = -1*np.ones(dtype=np.int32, shape=(1,T_i))
        tmp[0, :] = data_set[0][i][0, X_order:]
        data_S.append( tmp )
        
    #---inference
    inf_states = inference(model_file, "gaussian", total_data_X)
    
    total_conf_matrix = confusion_matrix(data_S[0][0,:], inf_states[0][0,:])
    errors = [ data_S[0][0,:] != inf_states[0][0,:] ]
    
    for i in range(1, N):
        total_conf_matrix = total_conf_matrix + \
                        confusion_matrix(data_S[i][0,:], inf_states[i][0,:])
        
        errors.append( data_S[i][0,:] != inf_states[i][0,:] )        
            
    return (total_conf_matrix, errors)


