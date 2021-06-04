import numpy as np
import concurrent.futures
from gaussian_innovation import Gaussian_X
from scaled_backward_forward_backward_recursion import BFB
from partially_hidden_MC import PHMC, update_F_from_Xi

#------------------------------------------------------------------------------
#   INITIALIZATION
#------------------------------------------------------------------------------
## @fn
#  @brief
#
def run_EM_on_init_param(X_order, nb_regimes, data, initial_values, states, \
                         innovation, nb_iters_init):
    
    #----AR process initialization  
    if (innovation == "gaussian"):
        X_process = Gaussian_X(X_order, nb_regimes, data, initial_values)            
    else:
        print()
        print("ERROR: file EM_learning.py: the given distribution is not supported!")
        exit (1)
            
    #----PHMC process initialization
    PHMC_process = PHMC(nb_regimes)
            
    #----run EM nb_iters times
    return EM(X_process, PHMC_process, states, nb_iters_init, epsilon=1e-5, \
              log_info=False)
 

## @fn
#  @brief EM initialization procedure.
#   Parameters are initialized nb_init times.
#   For each initialization, nb_iters of EM is executed.
#   Finally, the set of parameters that yield maximum likelihood is returned.
#   when more than one sequence have been observed, initialization
#   procedure is executed on the 10 sequences randomly chosen.
#
#  @param nb_init Number of initializations
#  @nb_iters Number of EM iterations
#
#  @return The set of parameters having the highest likelihood value.
#   (ar_intercept, ar_coefficient, ar_sigma, Pi_, A_)
#
def random_init_EM(X_order, nb_regimes, data, initial_values, states, \
                   innovation, nb_init, nb_iters_init):
        
    print("********************EM initialization begins**********************")
    
    #------Data used in initialization procedure
    # initialization is performed on the first 10 sequences
    
    #------Runs EM nb_iters_init times on each initial values 
    output_params = []
    
    #### Begin OpenMP parallel execution     
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = []
        for _ in range(nb_init):
            futures.append( executor.submit(run_EM_on_init_param,  X_order, \
                                            nb_regimes, data, \
                                            initial_values, states, \
                                            innovation, nb_iters_init) )
             
        for f in concurrent.futures.as_completed(futures):
            output_params.append(f.result())           
    #### End OpenMP parallel execution    
        
    #------Find parameters that yields maximum likehood
    #maximum likeliked within [-inf, 0]
    print("==================LIKELIHOODS====================")
    max_ll = -1e300  
    max_ind = -1
    for i in range(nb_init):     
                
        if (output_params[i][0] > max_ll):
            max_ll = output_params[i][0]
            max_ind = i
        
    print("==================INITIAL VALUES OF PARAMETERS====================")
    print("------------------AR process----------------------")
    print("total_log_ll= ", output_params[max_ind][0])
    print("ar_coefficients=", output_params[max_ind][5])
    print("intercept=", output_params[max_ind][7])
    print("sigma=", output_params[max_ind][6])
    print()
    print("------------------Markov chain----------------------")
    print("Pi=", output_params[max_ind][2])
    print("A=", output_params[max_ind][1])
    
    print("*********************EM initialization ends***********************")
    
    return (output_params[max_ind][7], output_params[max_ind][5], \
            output_params[max_ind][6], output_params[max_ind][2], \
            output_params[max_ind][1])

#------------------------------------------------------------------------------
#   LEARNING 
#------------------------------------------------------------------------------
## @fn
#  @brief
#
#  @param X_order
#  @param nb_regimes
#  @param data
#  @param initial_values
#  @param states
#  @param innovation
#  @param nb_iters
#  @param epsilon
#
#  @return 
#  #nb_iters=500, epsilon=1e-6, nb_init=10, nb_iters_init=5
#
def hmc_lar_parameter_learning(X_order, nb_regimes, data, initial_values, \
                             states, innovation, nb_iters=500, epsilon=1e-6, \
                             nb_init=10, nb_iters_init=5):
    
    #---------------AR process initialization  
    if (innovation == "gaussian"):
        X_process = Gaussian_X(X_order, nb_regimes, data, initial_values)        
    elif (innovation == "gamma"):
        X_process = Gamma_X(X_order, nb_regimes, data, initial_values)        
    else:
        print()
        print("ERROR: file EM_learning.py: the given distribution is not supported!")
    
    #---------------PHMC process initialization
    PHMC_process = PHMC(nb_regimes)
    
    #---------------psi estimation
    X_process.estimate_psi_MLE()
        
    #---------------EM learning
    (ar_inter, ar_coef, ar_sigma, Pi_, A_) = random_init_EM(X_order, \
                        nb_regimes, data, initial_values, states, innovation, \
                        nb_init, nb_iters_init)
    X_process.set_parameters(ar_inter, ar_coef, ar_sigma)
    PHMC_process.set_parameters(Pi_, A_)
       
    #----EM running
    return EM(X_process, PHMC_process, states, nb_iters, epsilon)
    
  
## @fn
#  @brief
#
def run_BFB_s(X_process, PHMC_process, states_s, s):
    
    #LL a T_s x X_order matrix of likelihood with
    #LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ; \theta^{(X,z)})
    LL = X_process.total_likelihood_s(s)
    
    #run backward-forward-backward on the s^th sequence
    return BFB(X_process.nb_regime, LL, states_s, PHMC_process.A, \
               PHMC_process.Pi)  
    
## @fn
#  @brief
#
def M_X_step(X_process, list_Gamma):  
    conv = X_process.update_parameters(list_Gamma)
    return conv

## @fn
#  @brief
#
def M_Z_step(PHMC_process, F, list_Gamma):
    PHMC_process.update_parameters(F, list_Gamma)

##
#
def compute_norm(prev_estimated_param, PHMC_process, X_process):
        
    A = prev_estimated_param[1]
    Pi = prev_estimated_param[2]
    ar_coefficients = prev_estimated_param[5]
    ar_intercepts = prev_estimated_param[7]  
    Sigma = prev_estimated_param[6]
    
    norm = np.sum(np.abs(PHMC_process.A - A))
    norm = norm + np.sum(np.abs(PHMC_process.Pi - Pi))
    norm = norm + np.sum(np.abs(X_process.coefficients - ar_coefficients))
    norm = norm + np.sum(np.abs(X_process.intercept - ar_intercepts))
    norm = norm + np.sum(np.abs(X_process.sigma - Sigma))
    
    return norm

## @fn
#  @brief
#
#  @param X_process
#  @param states sequences, states[s] = sigma^{(s)} a line vector 1xT, 
#  with Sigma[0,t] = -1 if Z_t is latent otherwise
#  Sigma[0,t] in {0, 1, 2, ..., M-1}.
#  sigma^{(s)}_t in {0, 1, 2, ..., M-1} u {-1} and: \n
#       * sigma^{(s)}_t = -1  if Z^{(s)}_t is latent \n
#       * otherwise Z^{(s)}_t is observed 
#  @param nb_iters
#  @param epsilon
#  @param log_info
#
#  @return 
#
def EM (X_process, PHMC_process, states, nb_iters, epsilon, log_info=True):
    
    #nb observed sequences
    S = len(X_process.data)
    #nb regimes
    M = X_process.nb_regime
    
    #total_log_ll is in [-inf, 0]
    prec_total_log_ll = -1e300  
    
    #current/previous estimated parameters
    prev_estimated_param = (-np.inf,  PHMC_process.A, PHMC_process.Pi, [], [], \
               X_process.coefficients, X_process.sigma, X_process.intercept,  \
               X_process.psi)
    
    curr_estimated_param = prev_estimated_param
        
    #--------------------------------------------nb_iters of EM algorithm
    for ind in range(nb_iters):
    
        #matrix of transition frequency
        F = np.zeros(dtype=np.float64, shape=(M, M))
        #
        list_Gamma = [{} for _ in range(S)]
        #
        list_Alpha = [{} for _ in range(S)]
        #
        total_log_ll = 0.0
            
        #----------------------------begin E-step  
        #### begin OpenMP parallel execution
        # list of tasks, one per sequence
        list_figures = [{} for _ in range(S)]
        with concurrent.futures.ThreadPoolExecutor() as executor:
            #run backward forward backward algorithm on each equence
            for s in range(S):
                list_figures[s] = executor.submit(run_BFB_s, X_process, \
                                                  PHMC_process, states[s], s)
                
            for s in range(S):
                #run BFB on the s^{th} sequence
                (log_ll_s, Xi_s, Gamma_s, Alpha_s) = list_figures[s].result()
                #update transition frequency matrix
                F = update_F_from_Xi(F, Xi_s)           
                #Gamma and Alpha probabilities computed on the s^th sequence
                list_Gamma[s] = Gamma_s
                list_Alpha[s] = Alpha_s          
                #total log-likelihood over all observed sequence
                total_log_ll = total_log_ll + log_ll_s  
                
        #### end OpenMP parallel execution
        #----------------------------end E-step
        
        #----------------------------begin M-steps 
        #### begin OpenMP parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor: 
            #-----------M-Z step
            task1 = executor.submit(M_Z_step, PHMC_process, F, list_Gamma)
            #-----------M-X step
            task2 = executor.submit(M_X_step, X_process, list_Gamma)
            
            task1.result()
            task2.result()
            
        #### end OpenMP parallel execution
        #----------------------------end M-steps 
        
        #-----------------------begin EM stopping condition        
        delta_log_ll = total_log_ll - prec_total_log_ll   
        
        abs_of_diff = compute_norm(prev_estimated_param, PHMC_process, X_process) 
        
        curr_estimated_param = ((total_log_ll + X_process.init_val_ll()),  \
               PHMC_process.A, PHMC_process.Pi, list_Gamma, list_Alpha,  \
               X_process.coefficients, X_process.sigma, X_process.intercept,  \
               X_process.psi)
    
        if(abs_of_diff < epsilon):  #convergence           
            #LOG-info
            if(log_info):
                print("--------------EM CONVERGENCE------------")
                print("#EM_converges after {} iterations".format(ind+1))
                print("schift in parameter = {}".format(abs_of_diff))
            break       
        else:
            #LOG-info
            if(log_info):
                print("iterations = {}".format(ind+1))
                print("schift in parameter = {}".format(abs_of_diff))
        
            #update prec_total_log_ll
            prec_total_log_ll = total_log_ll
        
            #update prev_estimated_param
            prev_estimated_param = curr_estimated_param
        #-----------------------end EM stopping condition  
                         
    return curr_estimated_param
      


## @fn
#  @brief
#    
def compute_ll(X_process, list_Gamma):
    
    log_ll = 0.0
    
    for s in range(len(list_Gamma)):
        
        LL_s = X_process.total_likelihood_s(s)
        
        for t in range(list_Gamma[s].shape[0]):
            
            log_ll = log_ll + np.log(np.sum( LL_s[t, :] * \
                                            list_Gamma[s][t, :]))
            
    return log_ll

    
   
