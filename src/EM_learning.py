import numpy as np
import concurrent.futures
from gaussian_innovation import Gaussian_X
from scaled_backward_forward_backward_recursion import BFB
from partially_hidden_MC import PHMC, update_F_from_Xi, update_MC_parameters



################################ BEGIN UTILS FUNCTIONS

## @fn compute_ll
#  @brief Another way to compute log-likelihood ll.
#   Likelihood of X_t can be computed as the weighted sum of the likehood within 
#   each regime:
#   P(X_t|past_values) = \sum_k P(X_t|past_values, Z_t=k) * P(Z_t=k).
#   Note that weights are regimes' probability at timestep t provided in
#   parameter list_Gamma.
#   In the end ll is equal to the sum of log( P(X_t|past_values) ) over all
#   timestep for all time series.  
#      
def compute_ll(X_process, list_Gamma):
    
    log_ll = 0.0
    
    for s in range(len(list_Gamma)):
        
        LL_s = X_process.total_likelihood_s(s)
        
        for t in range(list_Gamma[s].shape[0]):
            
            log_ll = log_ll + np.log(np.sum( LL_s[t, :] * \
                                            list_Gamma[s][t, :]))
            
    return log_ll + X_process.init_val_ll()


## @fn
#  @brief Run BFB algorithm on the s^{th} sequence.
#
#  @return BFB outputs and sequence index which is used in main process 
#   to properly collect the results of different child processes.
#
def run_BFB_s(M, LL, A, Pi, states_s, s):
        
    #run backward-forward-backward on the s^th sequence
    return (s,  BFB(M, LL, states_s, A, Pi)  )
    
    
## @fn
#  @brief Run the step M_X of EM in which PHMC_process parameters are updated.
#
def M_Z_step(F, list_Gamma):
    return update_MC_parameters(F, list_Gamma)


## @fn compute_norm
#  @brief Compute the L1-norm of the difference between previous estimation and
#   current estumation of parameters.
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

################################ END UTILS FUNCTIONS


################################ BEGIN INITIALIZATION

## @fn
#  @brief Initialize model parameters randomly then run nb_iters_init steps of 
#   EM and return the resulting estimation of parameters.
#
def run_EM_on_init_param(X_order, nb_regimes, data, initial_values, states, \
                         innovation, nb_iters_init):
    
    # re-seed the pseudo-random number generator.
    # This guarantees that each process/thread runs this function with
    # a different seed.
    np.random.seed()
    
    #----AR process initialization  
    if (innovation == "gaussian"):
        X_process = Gaussian_X(X_order, nb_regimes, data, initial_values, \
                               states, method="rand")              
    else:
        print()
        print("ERROR: file EM_learning.py: the given distribution is not supported!")
        exit (1)
            
    #----PHMC process initialization
    PHMC_process = PHMC(nb_regimes)
            
    #----run EM nb_iters_init times
    return EM(X_process, PHMC_process, states, nb_iters_init, epsilon=1e-5, \
              log_info=False)
 

## @fn
#  @brief EM initialization procedure.
#   Parameters are initialized nb_init times.
#   For each initialization, nb_iters_init of EM is executed.
#   Finally, the set of parameters that yield maximum likelihood is returned.
#
#  @param nb_init Number of initializations.
#  @nb_iters_init Number of EM iterations.
#
#  @return The set of parameters having the highest likelihood value.
#   (ar_intercept, ar_coefficient, ar_sigma, Pi_, A_).
#
def random_init_EM(X_order, nb_regimes, data, initial_values, states, \
                   innovation, nb_init, nb_iters_init):
        
    print("********************EM initialization begins**********************")  
    
    #------Runs EM nb_iters_init times on each initial values 
    output_params = []
    
    #### Begin multi-process parallel execution    
    nb_workers = min(10, nb_init)
    with concurrent.futures.ProcessPoolExecutor(max_workers=nb_workers) as executor:
        futures = []
        for _ in range(nb_init):
            futures.append( executor.submit(run_EM_on_init_param,  X_order, \
                                            nb_regimes, data, \
                                            initial_values, states, \
                                            innovation, nb_iters_init) )
             
        for f in concurrent.futures.as_completed(futures):
            output_params.append(f.result())            
    #### End multi-process parallel execution    
        
    #------Find parameters that yields maximum likehood
    #maximum likeliked within [-inf, 0]
    ##print("==================LIKELIHOODS====================")
    max_ll = -1e300  
    max_ind = -1
    for i in range(nb_init):             
        print(output_params[i][0])        
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

################################ END INITIALIZATION


################################ BEGIN  LEARNING 

## @fn
#  @brief Learn PHMC-LAR model parameters by EM algortihm.
#
#  @param X_order Autoregressive order, must be positive.
#  @param nb_regimes Number of switching regimes.
#
#  @param data List of length S, where S is the number of observed
#  time series. data[s], the s^th time series, is a column vector
#  T_sx1 where T_s denotes its size starting at timestep t = order + 1 included.
#
#  @param initial_values List of length S. initial_values[s] is a column vector 
#   orderx1 of initial values associated with the s^th time series.
#
#  @param states List of S state sequences. states[s] is a line vector 1xT: \n
#    * if states[s][0,t] = -1 then Z^{(s)}_t is latent.
#    * otherwise states[s][0,t] is in {0, 1, 2, ..., M-1} and Z^{(s)}_t
#      is observed to be equal to states[s][0,t]. M is the number of regimes.
#
#  @param innovation Law of model error terms, only 'gaussian' noises are supported  
#  @param nb_iters Maximum number of EM iterations.
#  @param epsilon Convergence precision. EM will stops when the shift 
#   in parameters' estimate between two consecutive iterations is less than 
#   epsilon. L1-norm was used.
#
#  @return Parameters' estimation computed by EM algorithm.
#   * log_ll 
#   * transition matrix
#   * initial law
#   * list_Gamma
#   * list_Alpha
#   * ar_coefficients 
#   * standard deviation
#   * intercept
#   * initial law parameters
#
def hmc_lar_parameter_learning(X_order, nb_regimes, data, initial_values, \
                             states, innovation, init_method="rand", \
                             nb_iters=500, epsilon=1e-6, \
                             nb_init=10, nb_iters_init=5):
    
    #---------------AR process initialization  
    if (innovation == "gaussian"):
        X_process = Gaussian_X(X_order, nb_regimes, data, initial_values, \
                               states, init_method)             
    else:
        print()
        print("ERROR: file EM_learning.py: the given distribution is not supported!")
    
    #---------------PHMC process initialization
    PHMC_process = PHMC(nb_regimes)
    
    #---------------psi estimation
    X_process.estimate_psi_MLE()
        
    #---------------several random starting values
    (ar_inter, ar_coef, ar_sigma, Pi_, A_) = random_init_EM(X_order, \
                        nb_regimes, data, initial_values, states, innovation, \
                        nb_init, nb_iters_init)
    X_process.set_parameters(ar_inter, ar_coef, ar_sigma)
    PHMC_process.set_parameters(Pi_, A_)
        
    #---------------run EM
    return EM(X_process, PHMC_process, states, nb_iters, epsilon)
    
  
## @fn
#  @brief
#
#  @param X_process Object of class RSARM.
#  @param PHMC_process Object of class PHMC.
#  @param states
#  @param nb_iters
#  @param epsilon
#  @param log_info
#
#  @return Parameters' estimation computed by EM algorithm.
#
def EM (X_process, PHMC_process, states, nb_iters, epsilon, log_info=True):
    
    #nb observed sequences
    S = len(X_process.data)
    #nb regimes
    M = X_process.nb_regime
    
    #total_log_ll is in [-inf, 0]
    prec_total_log_ll = -1e300  
    
    #current/previous estimated parameters
    prev_estimated_param = (-np.inf,  PHMC_process.A.copy(), \
                            PHMC_process.Pi.copy(), [], [], \
                            X_process.coefficients.copy(), \
                            X_process.sigma.copy(), X_process.intercept.copy(),  \
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
        #### begin multi-process parallel execution
        # list of tasks, one per sequence
        list_futures = []
        with concurrent.futures.ProcessPoolExecutor(max_workers=32) as executor:
            #run backward forward backward algorithm on each equence
            for s in range(S):
                list_futures.append( executor.submit(run_BFB_s, M, \
                                     X_process.total_likelihood_s(s), \
                                     prev_estimated_param[1], \
                                     prev_estimated_param[2], states[s], s) )
                
            #collect the results as tasks complete
            for f in concurrent.futures.as_completed(list_futures):
                #results of the s^{th} sequence
                (s, (log_ll_s, Xi_s, Gamma_s, Alpha_s)) = f.result()
                #update transition frequency matrix
                F = update_F_from_Xi(F, Xi_s)           
                #Gamma and Alpha probabilities computed on the s^th sequence
                list_Gamma[s] = Gamma_s
                list_Alpha[s] = Alpha_s          
                #total log-likelihood over all observed sequence
                total_log_ll = total_log_ll + log_ll_s                  
        #### end multi-process parallel execution
        #----------------------------end E-step
        
        #----------------------------begin M-steps 
        #### begin multi-process parallel execution
        with concurrent.futures.ProcessPoolExecutor(max_workers=1) as executor: 
            #------M-Z step
            task = executor.submit(M_Z_step, F, list_Gamma)
            
            #------M-X step
            X_process.update_parameters(list_Gamma)
 
            #collect result of task and update Theta_Z
            (Pi_, A_) = task.result()
            PHMC_process.set_parameters(Pi_, A_)           
        #### end multi-process parallel execution
        #----------------------------end M-steps 
        
        #-----------------------begin EM stopping condition        
        delta_log_ll = total_log_ll - prec_total_log_ll   
        
        abs_of_diff = compute_norm(prev_estimated_param, PHMC_process, X_process) 
        
        curr_estimated_param = ((total_log_ll + X_process.init_val_ll()),  \
                                PHMC_process.A.copy(), PHMC_process.Pi.copy(), \
                                list_Gamma, list_Alpha,  \
                                X_process.coefficients.copy(), \
                                X_process.sigma.copy(), X_process.intercept.copy(),  \
                                X_process.psi)
    
        if(np.isnan(abs_of_diff)):  #EM stops with a warning, ADDED 21/06/13
            #LOG-info
            if(log_info):
                print("--------------EM stops with a warning------------")
                print("At iteration {}, NAN values encountered in the estimated parameters".format(ind+1))
                print("log_ll = {}".format(total_log_ll))
                print("delta_log_ll = {}".format(delta_log_ll))
                print("schift in parameter = {}".format(abs_of_diff))
                print("PARAMETERS AT ITERATION {} ARE RETURNED".format(ind))
            return prev_estimated_param
        
        if(abs_of_diff < epsilon):  #convergence           
            #LOG-info
            if(log_info):
                print("--------------EM CONVERGENCE------------")
                print("#EM_converges after {} iterations".format(ind+1))
                print("log_ll = {}".format(total_log_ll))
                print("delta_log_ll = {}".format(delta_log_ll))
                print("schift in parameter = {}".format(abs_of_diff))
            break       
        else:
            #LOG-info
            if(log_info):
                print("iterations = {}".format(ind+1))
                print("log_ll_alpha = {}".format(total_log_ll))
                print("delta_log_ll = {}".format(delta_log_ll))
                print("schift in parameter = {}".format(abs_of_diff))
        
            #update prec_total_log_ll
            prec_total_log_ll = total_log_ll
        
            #update prev_estimated_param
            prev_estimated_param = curr_estimated_param
        #-----------------------end EM stopping condition  
             
    """
    #another way to compute log_likelihood
    print("another way to compute log_likelihood = ", \
                                          compute_ll(X_process, list_Gamma) )
    """
            
    return curr_estimated_param
      
   
    


    
   
