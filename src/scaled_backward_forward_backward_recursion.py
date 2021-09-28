###############################################################################
## @module BFB 
#  @brief This module impements Backward Forward Backward recursion algorithm.
#   In this version quantities tau, alpha, beta are normalized in order to 
#   avoid underflow. 
#
#  Let us consider a regime switching AR process X modeled by a probability 
#  distribution parametrized by \theta^{(X)}, (x_1, x_2, ..., x_T) 
#  one realization of X.\n
#  Regimes are modeled by a partially hidden Markov chain process 
#  Z having M possible states, parametrized by \theta^{(Z)} = (A, Pi) where, 
#  A is the matrix of transition probabilities and Pi the initial state 
#  probability Pi.
#  Let (Z_1, Z_2, ..., Z_T) the states associated with (x_1, x_2, ..., x_T) .\n
#  Let sigma^{(s)}_t in {0, 1, 2, ..., M-1} u {-1} with: \n
#       * sigma^{(s)}_t = -1  if Z^{(s)}_t is latent \n
#       * otherwise Z^{(s)}_t is observed 
#
#  BFB computes joint probabilities Xi_t(i,j) = P(Z_t=i, Z_{t+1}=j | X, Sigma, Theta),
#  the probability to observe the state i at time t (marginal probabibilies)
#  gamma_t = P(Z_t=i | X, Sigma, Theta) and the likelihood of observation data P(X).
#
#  BFB is a generalization of the well-known Forward Backward 
#  recursion in which the process is completely hidden.
#  
#  What happens when the process is completely observed ????
#
###############################################################################

import numpy as np
import math

"""****************BEGIN*********************************************"""
## @fn BFB
#  @brief This function runs BFB algorithm.
#
#  @param M The number of states.
#  @param LL The likelihood matrix of dimension TxM where: \n
#  * LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ; \theta^{(X,z)}), where d is AR
#    process order. \n
#  * \theta^{(X,z)} is the set of parameters related to the z^th regime.
#  * order is the number of past values of X_t used in its prediction.
#
#  @param Sigma 1xT array (row vector), with Sigma[0,t] = -1 if Z_t is latent 
#   otherwise. Sigma[0,t] in {0, 1, 2, ..., M-1}.
#  
#  @param A Matrix MxM, A[i,j] = a_{i,j} = P(Z_t=j|Z_{t-1}=i).
#  @param Pi Row vector 1xM, initial state probabilities, Pi[0,k] = P(Z_0 = k).
#
#  @return the log-likelihood of observation data P(X|Sigma, Theta).
#  @return the joint probabilities Xi_t = P(Z_t=i, Z_{t+1}=j | X, Sigma, Theta).
#  @return the probability to observe the state i at time t gamma_t = P(z_t=i | X, Sigma, Theta).
#  
def BFB(M, LL, Sigma, A, Pi):
            
    #----------probabilities P(sigma_t|sigma_{t-1}) computing
    prob_sigma = compute_prob_sigma(M, Sigma, A, Pi)
                  
    #----------BFB algorithm begins
    #B1-step
    Tau_tilde = first_backward_step(M, Sigma, A, prob_sigma)
    #F-step
    (Alpha_tilde, C_t) = forward_step(M, LL, Sigma, A, Pi, Tau_tilde, \
                                        prob_sigma)
    #B2-step
    Beta_tilde = second_backward_step( M, LL, Sigma, A, Pi, Tau_tilde, C_t, \
                                         prob_sigma)
    #Xi computing
    Xi = compute_Xi(LL, Sigma, Tau_tilde, A, Pi, Alpha_tilde, Beta_tilde, \
                    prob_sigma)
    #Gamma computing
    Gamma = compute_gamma(Xi)
    """
    #another way to compute gamma
    Gamma = compute_gamma_bis(Alpha_tilde, Beta_tilde, C_t)
    """
    #ll computing
    log_ll = likelihood(C_t)
    
    
    return (log_ll, Xi, Gamma, Alpha_tilde)


## @fn
#
#
def compute_prob_sigma(M, Sigma, A, Pi):
    
    T = np.shape(Sigma)[1]
    prob_sigma = np.zeros(dtype=np.float64, shape=T)
    
    #---first time-step
    class_ = Sigma[0, 0]  

    if(class_ == -1):              #S_1 is hidden
        prob_sum = np.sum(Pi[0, :]) 
    else:                         #S_1 is observed      
        prob_sum = Pi[0, class_]
          
    prob_sigma[0] = prob_sum + np.finfo(0.).tiny
                           
    #---other time-step
    for t in range(1, T):
                
        if(Sigma[0, t] == -1):
            set_sigma_t = [ind for ind in range(M)]
        else:
            set_sigma_t = [ Sigma[0, t] ]
            
        if(Sigma[0, t-1] == -1):
            set_sigma_t_1 = [ind for ind in range(M)]
        else:
            set_sigma_t_1 = [ Sigma[0, t-1] ]
            
        prob_sum = 0.0        
        for i in set_sigma_t_1:
            for j in set_sigma_t:
                prob_sum = prob_sum + A[i, j]
                   
        prob_sigma[t] = prob_sum + np.finfo(0.).tiny
    
        
    return prob_sigma


"""****************BEGIN (a) *********************************************"""
## @fn first_backward_step
#  @brief This function runs the first backward step of BFB.
#
#  @param M The number of states.
#  @param Sigma Row vector 1xT, with Sigma[0,t] = -1 if Z_t is latent otherwise
#   Sigma[0,t] in {0, 1, 2, ..., M-1}.
#
#  @param A Matrix MxM, A[i,j] = a_{i,j} = P(Z_t=j|Z_{t-1}=i).
#
#  @parma prob_sigma Array of length T containing probabilities
#   P(\sigma_t | sigma_{t-1}; Theta)
#  
#  @return A matrix TxM Tau with Tau[t,i] = P(......).
#  
def first_backward_step(M, Sigma, A, prob_sigma):
    
    #initialization
    T = np.shape(Sigma)[1]
    Tau_tilde = np.zeros(dtype=np.float64, shape=(T, M))
    
    #----------base case, t = T-1
    Tau_tilde[T-1, :] = np.ones(dtype=np.float64, shape=M) / prob_sigma[T-1]
    
    #----------recursive case
    for t in range(T-2, -1, -1): 
        
        class_ = Sigma[0, t+1]
        
        for z in range(M):     
                        
            if(class_ == -1):  #Sigma_{t+1} is latent  
                tau_t_z = np.sum(Tau_tilde[t+1, :] * A[z, :]) / prob_sigma[t]    
                
            else:   #Sigma_{t+1} is observed 
                tau_t_z = (Tau_tilde[t+1, class_] * A[z, class_]) / \
                                prob_sigma[t]                   
                                     
            #assertion: valid value domain
            assert(not math.isnan(tau_t_z))
            assert(tau_t_z >= 0.) 
                      
            #product of values within [0,1[ tends fastly to 0
            Tau_tilde[t, z] = tau_t_z + np.finfo(0.).tiny
                                
    return Tau_tilde


"""****************BEGIN (b) AND (c)****************************************"""
## @fn forward_step
#  @brief This function runs the forward step of BFB.
#
#  @param M The number of states.
#  @param LL The likelihood matrix of dimension TxM where: \n
#  * LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ; \theta^{(X,z)}), where d is AR
#    process order. \n
#  * \theta^{(X,z)} is the set of parameters related to the z^th regime.
#  * order is the number of past values of X_t used in its prediction.
#
#  @param Sigma Row vector 1xT, with Sigma[0,t] = -1 if Z_t is latent otherwise
#   Sigma[0,t] in {0, 1, 2, ..., M-1}.
#  
#  @param A Matrix MxM, A[i,j] = a_{i,j} = P(Z_t=j|Z_{t-1}=i).
#  @param Pi Row vector 1xM, initial state probabilities, Pi[0,k] = P(Z_0 = k).
#
#  @param Tau_tilde Matrix TxM 
#  @parma prob_sigma Array of length T containing probabilities
#   P(\sigma_t | sigma_{t-1}; Theta)
#  
#
#  @return 
#   * A matrix TxM Alpha with Alpha[t,i] = P(......). \n
#     z not in Sigma_t => Alpha[t, z] = 0
#   * T length array C_t, with 
#     C_t[t] = P(X_t=x_t | X_{1-p}^{t-1}, Sigma, Theta)
#  
def forward_step(M, LL, Sigma, A, Pi, Tau_tilde, prob_sigma):
    
    #initialization
    T = np.shape(Sigma)[1]
    Alpha_tilde = np.zeros(dtype=np.float64, shape=(T, M))
    C_t = np.zeros(dtype=np.float64, shape=T)
    
    #----------base case of induction, t = 0
    class_ = Sigma[0, 0]
    
    #---compute C_1
    if(class_ == -1):   #latent state
        C_t[0] = np.sum(LL[0, :] * Pi[0, :])
    else:               #observed state
        C_t[0] = LL[0, class_] * Pi[0, class_]        
    C_t[0] = C_t[0] + np.finfo(0.).tiny
        
    #---compute Alpha_tilde_1
    for z in range(M):      
        
        aux = (LL[0, z] * Pi[0, z]) / C_t[0]        
        if(class_ == -1):   #latent state
            alpha_0_z = aux * \
                    ( Tau_tilde[0, z] / np.sum(Tau_tilde[0, :] * Pi[0, :]) )                         
        else:               #observed state
            alpha_0_z = aux * (Tau_tilde[0, z] / Tau_tilde[0, class_]) / \
                            (Pi[0, class_] + np.finfo(0.).tiny)
        
        #assertion: valid value domain
        assert(not math.isnan(alpha_0_z))
        assert(alpha_0_z >= 0.) 
                
        Alpha_tilde[0, z] = alpha_0_z
                
    #----------recursive case
    for t in range(1, T):
        
        class_ = Sigma[0, t-1]
        class_t = Sigma[0, t]
        
        #---compute C_t
        if(class_t == -1):
            set_sigma_t = [ind for ind in range(M)]
        else:
            set_sigma_t = [class_t]
            
        if(class_ == -1):
            set_sigma_t_1 = [ind for ind in range(M)]
        else:
            set_sigma_t_1 = [class_]
        
        tmp_sum = 0.0
        for i in set_sigma_t_1:
            for j in set_sigma_t:
                tmp_sum = tmp_sum + (LL[t, j] * Alpha_tilde[t-1, i] * A[i, j])   
                        
        C_t[t] = tmp_sum + np.finfo(0.).tiny
        
        #---compute Alpha_tilde_t
        for z in range(M):
                       
            if(class_t == -1 or class_t == z):  #z in Sigma_t
                                
                if (class_ == -1):  #Sigma_{t-1} is latent                    
                    alpha_t_z = LL[t, z] * \
                                np.sum( Alpha_tilde[t-1, :] * A[:, z] * \
                                (Tau_tilde[t, z] / Tau_tilde[t-1, :]) ) /   \
                                (C_t[t] * prob_sigma[t-1])
                             
                else:    #Sigma_{t-1} is observed 
                    alpha_t_z = LL[t, z] * Alpha_tilde[t-1, class_] * \
                                        A[class_, z] * \
                                (Tau_tilde[t, z] / Tau_tilde[t-1, class_]) / \
                                (C_t[t] * prob_sigma[t-1])
                                                                
            else:  #z not in Sigma_t
                alpha_t_z = 0.0
                            
            #assertion: valid value domain
            assert(not math.isnan(alpha_t_z))
            assert(alpha_t_z >= 0.)
            
            Alpha_tilde[t, z] = alpha_t_z
                                  
        
    return (Alpha_tilde, C_t)

"""**************BEGIN (d)************************************************"""
## @fn second_backward_step
#  @brief This function runs the second backward step of BFB.
#
#  @param M The number of states.
#  @param LL The likelihood matrix of dimension TxM where: \n
#  * LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ; \theta^{(X,z)}), where d is AR
#    process order. \n
#  * \theta^{(X,z)} is the set of parameters related to the z^th regime.
#  * order is the number of past values of X_t used in its prediction.
#
#  @param Sigma Row vector 1xT, with Sigma[0,t] = -1 if Z_t is latent otherwise
#   Sigma[0,t] in {0, 1, 2, ..., M-1}.
#  
#  @param A Matrix MxM, A[i,j] = a_{i,j} = P(Z_t=j|Z_{t-1}=i).
#  @param Pi Row vector 1xM, initial state probabilities, Pi[0,k] = P(Z_0 = k).
#
#  @param Tau_tilde Matrix TxM 
#  @param C_t
#  @parma prob_sigma Array of length T containing probabilities
#   P(\sigma_t | sigma_{t-1}; Theta)
#
#  @return A matrix TxM Beta with Beta[t,i] = P(......). \n
#  z not in Sigma_t => Beta[t, z] = 0
#
def second_backward_step(M, LL, Sigma, A, Pi, Tau_tilde, C_t, prob_sigma):
    
    #initialization
    T = np.shape(Sigma)[1]
    Beta_tilde = np.zeros(dtype=np.float64, shape=(T, M))
    
    #----------base case of induction, t = T-1
    Beta_tilde[T-1, :] = np.ones(dtype=np.float64, shape=M) / C_t[T-1]
    
    #----------recursion
    for t in range(T-2, -1, -1):
        
        class_ = Sigma[0, t+1]
        class_t = Sigma[0, t]
        
        for z in range(M):
                        
            if(class_t == -1 or class_t == z):  #z in Sigma_t
                    
                if (class_ == -1):  #Sigma_{t+1} is latent                     
                    tmp_sum = 0.0
                    for i in range(M):                   
                        aux1_ = Beta_tilde[t+1, i] * LL[t+1, i] * A[z, i] * \
                                    (Tau_tilde[t+1, i] / Tau_tilde[t, z])   
                        tmp_sum = tmp_sum + aux1_ 
                        
                    aux2_ = C_t[t] * prob_sigma[t] 
                    beta_t_z = min(1e300, tmp_sum/aux2_)
                                                                                                                
                else:      #Sigma_{t+1} is observed
                    aux1_ = LL[t+1, class_] * A[z, class_] * \
                            (Tau_tilde[t+1, class_] / Tau_tilde[t, z]) / C_t[t] 
                    aux2_ = Beta_tilde[t+1, class_] / prob_sigma[t]
                    beta_t_z = min(1e300, aux1_*aux2_)
                                                     
            else:  #z not in Sigma_t
                beta_t_z = 0.0
                         
            #assertion: valid value domain
            assert(not math.isnan(beta_t_z))
            assert(beta_t_z >= 0.)
            
            Beta_tilde[t, z] = beta_t_z
            
            
    return Beta_tilde


"""**************BEGIN (e)************************************************"""
## @fn log_likelihood
#  @brief This function computes the likelihood of observed data.
#
#  @param C_t T length array with 
#   C_t[t] = P(X_t=x_t | X_{1-p}^{t-1}, Sigma, Theta)
#
#  @return The log likelihood P(X|Sigma, Theta) (real value).
#  
def likelihood(C_t):
    
    T = C_t.shape[0]
    
    log_ll = 0.0    
    for t in range(T):
        log_ll = log_ll + np.log(C_t[t] + np.finfo(0.).tiny)
        
    #assertion: value domain
    assert(not math.isinf(log_ll))
    assert(not math.isnan(log_ll))
        
    return log_ll


## @fn compute_gamma_bis
#  @brief This function compute gamma probabilities from Alpha_tilde, 
#   Beta_tilde and C_t.
#  
#  @param Alpha_tilde
#  @param Beta_tilde
#  @param C_t
#
#  @return A matrix TxM Gamma with Gamma[t,i] = P(Z_t=z | X, Sigma, Theta).
#
def  compute_gamma_bis(Alpha_tilde, Beta_tilde, C_t):
      
    (T, M) = Alpha_tilde.shape
    Gamma = np.zeros(dtype=np.float64, shape=(T, M))
    
    for t in range(T):
        for z in range(M):
            Gamma[t, z] = Alpha_tilde[t, z] * Beta_tilde[t, z] * C_t[t]
            
        #normalization: Gamma[t, :] are in [0,1] and sums to one
        Gamma[t, :] = Gamma[t, :] / np.sum(Gamma[t, :])
                  
    return Gamma


## @fn compute_gamma
#  @brief This function compute gamma probabilities from Xi probabilities.
#  
#  @param Xi
#
#  @return A matrix TxM Gamma with Gamma[t,i] = P(Z_t=i | X, Sigma, Theta).
#
def  compute_gamma(Xi):
      
    (T, M, M) = Xi.shape
    Gamma = np.zeros(dtype=np.float64, shape=(T, M))
    
    #---first time-step
    for i in range(M):
        Gamma[0, i] = np.sum(Xi[1, i, :])
        
    #normalization: Gamma[0, :] are in [0,1] and sums to one
    Gamma[0, :] = Gamma[0, :] / np.sum(Gamma[0, :])
    
    #---other time-steps
    for t in range(1, T):
        for j in range(M):
            Gamma[t, j] = np.sum(Xi[t, :, j])
            
        #normalization: Gamma[t, :] are in [0,1] and sums to one
        Gamma[t, :] = Gamma[t, :] / np.sum(Gamma[t, :])
        
    return Gamma


"""**************BEGIN (f)************************************************"""
## @fn compute_Xi
#  @brief This function runs the second backward step of BFB.
#  
#  @param LL The likelihood matrix of dimension TxM where: \n
#  * LL[t,z] = g(x_t | x_{t-order}^{t-1}, Z_t=z ; \theta^{(X,z)}), where d is AR
#    process order. \n
#  * \theta^{(X,z)} is the set of parameters related to the z^th regime.
#  * order is the number of past values of X_t used in its prediction.
#
#  @param Sigma Row vector 1xT, with Sigma[0,t] = -1 if Z_t is latent otherwise
#   Sigma[0,t] in {0, 1, 2, ..., M-1}.
#   Paramètre non utilisé pour le moment mais sera utilisé dans la version
#   itermédiaire i.e. Z_t in {1, 2} seulment deux états plausible pour z_t
#
#  @param A Matrix MxM, A[i,j] = a_{i,j} = P(Z_t=j|Z_{t-1}=i).
#  @param Pi Row vector 1xM, initial state probabilities, Pi[0,k] = P(Z_0 = k).
#  Paramètre non utilisé pour le moment voir ci-dessus
#
#  @param Tau_tilde Matrix TxM, first backward propagation terms
#  @param Alpha_tilde Matrix TxM, foreward propagation terms
#  @param Beta_tilde Matrix TxM, second backward propagation terms
#
#  @parma prob_sigma Array of length T containing probabilities
#   P(\sigma_t | sigma_{t-1}; Theta)
#
#  @return A 3D matrix Xi with Xi[t, , ] a MxM matrix and  
#   Xi[t, i, j] = P(Z_{t-1}=i, Z_t=j | X, Sigma, \Theta). \n
#  If i not in Sigma_t or j not in Sigma_{t+1} => Xi[t, i, j] = 0. \n
#  Xi[1, , ] is a matrix MxM of zeros because Xi is not defined at the first
#  time-step.
#
def compute_Xi(LL, Sigma, Tau_tilde, A, Pi, Alpha_tilde, Beta_tilde, \
               prob_sigma):
    
    #initialization
    (T, M) = np.shape(Beta_tilde)
    Xi = np.zeros(dtype=np.float64, shape=(T, M, M))
        
    for t in range(1,T):
        
        #----set_sigma
        class_t = Sigma[0, t]
        class_t_1 = Sigma[0, t-1]
        
        if(class_t == -1):
            set_sigma_t = [ind for ind in range(M)]
        else:
            set_sigma_t = [class_t]
            
        if(class_t_1 == -1):
            set_sigma_t_1 = [ind for ind in range(M)]
        else:
            set_sigma_t_1 = [class_t_1]
        
        #---computing
        for i in set_sigma_t_1:
            for j in set_sigma_t:
                
                if(prob_sigma[t-1] != 0):
                    aux = Beta_tilde[t, j] * A[i, j] * LL[t, j] * \
                              (Alpha_tilde[t-1, i] / prob_sigma[t-1])
                    Xi[t, i, j] =  aux * (Tau_tilde[t, j] / Tau_tilde[t-1, i]) 
                                    
                #assertion: valid value domain
                assert(not math.isnan(Xi[t, i, j]))
                assert(Xi[t, i, j]  >= 0.)
    
        #if i is not in Sigma_{t-1} or j is not in Sigma_t then Xi[t, i, j] = 0     
                                 
    return Xi
    
