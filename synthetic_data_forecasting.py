#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
sys.path.append("src")

import pickle
import numpy as np

from forecasting_inference import forecasting, relabelling
from utils import compute_forecast_error



######################## UNLABELLED TEST-SET #################################
##  @fn
#   @brief Computes forecast task for the given model, on the given data
#
#   @param data_file
#   @param model_file PHMC-LAR model 
#   @param H Forecast horizons go from 1 to H
#
#  @return Forecast errors: RMSE, MAE, Bais
#
def forecast_task(data_file, model_file, H):
    
    #-----data_state_number_rho
    infile = open(data_file, 'rb')
    data = pickle.load(infile)
    infile.close()
    
    data_X = data[1]
        
    T = data_X[0].shape[0]
    N = len(data_X)
    order = 2

    list_init_values = []
    for s in range(N):
        list_init_values.append( data_X[s][(T-order):, ] )
    
    #------forecast
    predictions = forecasting(model_file, list_init_values, "gaussian", H)     
    
    #return forecat error    
    return compute_forecast_error(data[4], predictions, H)
    

##################### PARTIALLY LABELLED TEST-SET EXP1 ########################  
    
##  @fn 
#   @brief Performs forecast task for the given model, on the given data for 
#   which observation sequences are partially labelled
#
#   @param data_file
#   @param model_file PHMC-LAR model 
#   @param H Forecast horizons go from 1 to H
#   @param state_number, state_number[0] is the new number of the 1rst state
#   state_number, state_number[1] is the new number of the 2nd state
#
#   @return forecast errors
#
def forecast_task_bis(data_file, model_file, H, state_number=[]):
    
    #-----data
    infile = open(data_file, 'rb')
    data = pickle.load(infile)
    infile.close()
    
    data_X = data[1]
    
    T = data_X[0].shape[0]
    N = len(data_X)
    order = 2

    list_init_values = []
    list_states = []
    for s in range(N):       
        list_init_values.append( data_X[s][(T-order):, ] )       
        list_states.append(np.zeros(shape=(1,H), dtype=np.int32))
        list_states[s][0, :] = data[3][s][0,0:H] 
                    
    #------forecast
    if(len(state_number) == 0):  #no relabelling required    
        predictions = forecasting(model_file, list_init_values, "gaussian",\
                                  H, list_states)      
        
    else:    #regimes are relabelled before forecasting  
        relabelled_list_states = []
        for s in range(N):
            relabelled_list_states.append(-2*np.ones(shape=(1, H), \
                                                     dtype=np.int32))
            for i in range(H):
                relabelled_list_states[s][0,i] = \
                                             state_number[list_states[s][0,i]]
                                             
        predictions = forecasting(model_file, list_init_values, "gaussian",\
                                  H, relabelled_list_states) 
    
    return compute_forecast_error(data[4], predictions, H)


####################### TEST-SET LABELLING #################################### 

from synthetic_data_experiment_1 import create_partially_labelled_seq 

##  @fn 
#   @brief Creates partially labelled observations from the given forecast 
#   test data sets
#
def TEST_SET_create_partially_labelled_seq(T, N):
    
    data_file = "../simulated-data/hmc-lar2/data/TRAINING/data-set_train_T=" + \
                    str(T) + "_N=" + str(N)      
    output_dir = "../simulated-data/hmc-lar2/data/FORECASTING-TEST-SET/"
             
    #load test-set
    infile = open(data_file, 'rb')
    train_data = pickle.load(infile)
    infile.close()
        
    for P in [25, 50, 75, 100]:
        
        #labelled P% of observations
        res = create_partially_labelled_seq(train_data[3], nb_regimes=4, P=P)
        
        #save new labelled data-set
        output_file = output_dir + "data_set_train_T=" + str(T) + "_N=" + \
                        str(N) + "_P-test=" + str(P)
            
        outfile = open(output_file, 'wb') 
        pickle.dump( (train_data[0], train_data[1], train_data[2], res, \
                          train_data[4]), outfile)
        outfile.close()
