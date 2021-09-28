#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("src")

import pickle
import numpy as np

from forecasting_inference import inference
from utils import mean_percentage_error



############################ UNLABELLED TEST-SET ############################# 
##  @fn 
#   @brief Performs inference task for the given model, on the given data
#
#   @param data_file
#   @param model_file PHMC-LAR model 
#   @param state_number, state_number[0] is the new number of the 1rst state,
#    state_number[1] is the new number of the 2nd state.
#    The new numbers are the ones obtained after model training.
#
#   @return mean percentage of inference errors
#
def inference_task(data_file, model_file, state_number=[]):
    
    #-----test data
    infile = open(data_file, 'rb')
    test_data = pickle.load(infile)
    infile.close()
    
    #list of state sequences
    test_data_S = test_data[0]
    #list of observation sequences
    test_data_X = test_data[1]
    #list of initial values
    test_init_vals = test_data[2]
    
    N_s = len(test_data_S) 
    T_s = test_data_S[0].shape[1]    
    assert(N_s == 100 and T_s == 1000)
    
    #-----each sequence of observations and the corresponding initial values
    # are concatenated 
    test_total_X = []
    for s in range(N_s):
        test_total_X.append( np.vstack((test_init_vals[s], test_data_X[s])) )

    #-----inference 
    inf_states = inference(model_file, "gaussian", test_total_X)
        
    if(len(state_number) == 0):    #no relabelling required
        
        return mean_percentage_error(test_data_S, inf_states)
    
    else:   #regimes are relabelled before inference       
        
        relabelled_test_data_S = []
        for s in range(N_s):
            relabelled_test_data_S.append(-2*np.ones(shape=(1,T_s), \
                                                             dtype=np.int32))       
            for i in range(T_s):
                relabelled_test_data_S[s][0,i] = \
                                            state_number[test_data_S[s][0,i]]

        return mean_percentage_error(relabelled_test_data_S, inf_states)



##################### PARTIALLY LABELLED TEST-SET #############################  
##  @fn 
#   @brief Performs inference task for the given model, on the given data for
#    which observation sequences are partially labelled
#
#   @param data_file
#   @param model_file PHMC-LAR model 
#   @param state_number, state_number[0] is the new number of the 1rst state
#   state_number, state_number[1] is the new number of the 2nd state
#
#   @return mean percentage of inference errors
#
def inference_task_bis(data_file, model_file, state_number=[]):
    
    #-----test data
    infile = open(data_file, 'rb')
    test_data = pickle.load(infile)
    infile.close()
    
    #sequences 
    test_data_S = test_data[0]
    test_data_X = test_data[1]
    test_init_vals = test_data[2]
    test_data_S_P = test_data[3]
    
    N_s = len(test_data_S) 
    T_s = test_data_S[0].shape[1]    
    assert(N_s == 100 and T_s == 1000)
    
    test_total_X = []
    for s in range(N_s):
        test_total_X.append( np.vstack((test_init_vals[s], test_data_X[s])) )
    
    if(len(state_number) == 0):  #no relabelling required
        
        #inference
        inf_states = inference(model_file, "gaussian", test_total_X, \
                                   test_data_S_P)    
        return mean_percentage_error(test_data_S, inf_states)
        
    else:  #regimes ares relabelled before inference 
    
        relabelled_test_data_S = []
        relabelled_test_data_S_P = []
    
        for s in range(N_s):        
        
            relabelled_test_data_S.append(-2*np.ones(shape=(1,T_s), dtype=np.int32))   
            relabelled_test_data_S_P.append(-2*np.ones(shape=(1,T_s), dtype=np.int32))
        
            for i in range(T_s):
            
                #--test_data_S
                relabelled_test_data_S[s][0,i] = state_number[test_data_S[s][0,i]]
            
                #--test_data_S_P
                if(test_data_S_P[s][0,i] == -1):
                    relabelled_test_data_S_P[s][0,i] = -1
                else:
                    relabelled_test_data_S_P[s][0,i] = \
                                            state_number[test_data_S_P[s][0,i]]          
            
        #inference
        inf_states = inference(model_file, "gaussian", test_total_X, \
                               relabelled_test_data_S_P)
        return mean_percentage_error(relabelled_test_data_S, inf_states)
    

####################### TEST-SET LABELLING #################################### 
  
from synthetic_data_experiment import create_partially_labelled_seq 

##  @fn 
#   @brief Creates partially labelled observations from the given inference
#   test data sets
#
def TEST_SET_create_partially_labelled_seq(output_dir):
    
    data_file = "data/synthetic-data/data-set_test_T=1000_N=100" 
    #load test-set
    infile = open(data_file, 'rb')
    test_data = pickle.load(infile)
    infile.close()
        
    #labelling percentage     
    list_P_test = [0, 25, 50, 75, 100]
    
    for P_test in list_P_test:
        
        #labelled P% of observations
        res = create_partially_labelled_seq(test_data[0], 4, P_test)
        
        #save new labelled data-set
        output_file = output_dir + "data-set_test_T=1000_N=100_P-test=" + \
                        str(P_test)
        outfile = open(output_file, 'wb') 
        pickle.dump((test_data[0], test_data[1], test_data[2], res), outfile)
        outfile.close()
