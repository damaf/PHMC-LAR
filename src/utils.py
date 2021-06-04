import numpy as np

## @fn 
#  @brief Compute the mean percentage error between two partitions
#
#  @param data_S1 List of N states sequences, each is a 1xT arrays.
#   Reference labels.
#  @param data_S2 List of N states sequences, each is a 1xT arrays.
#   Inferred labels.
#
#  @return N-length array
#
def mean_percentage_error(data_S1, data_S2):
    
    #nb sequences
    N1 = len(data_S1)
    N2 = len(data_S2)  
    assert(N1 == N2) 
    
    mean_errors = -1 * np.ones(shape=N1, dtype=np.float64)
    
    for s in range(N1):
        
        #sequence length
        T1 = data_S1[s].shape[1]
        T2 = data_S2[s].shape[1]  
        assert(T1 == T2)        
    
        assert(np.sum(data_S1[s][0,:] < 0) == 0)
        assert(np.sum(data_S2[s][0,:] < 0) == 0)
                
        mean_errors[s] =  np.mean(data_S1[s][0,:] != data_S2[s][0,:])
               
    return mean_errors
