import numpy as np
from MF import Factor, Get_prediction_error
#Helper python script to interface with C++ functions

def CreateLatentVariables(n, g, k, sigma=0.02):
    U = np.random.randn(n, k) * sigma
    V = np.random.randn(g, k) * sigma
    return U, V

def FactorizeMatrix(X, U, V, neighbors, eta=0.005, lamb1=0.02, lamb2=0.001, num_epochs=200, known_indices=None):
    '''
    Factorizes the sparse matrix X into the product of two rank k matrices
    U and V using stochastic gradient descent.
    
    Args:
        X: An n x g, possibly sparse numpy matrix, where missing entries are indicated by np.nan values,
           where n represents the number of samples and g represents the number of genes, or items.
        U: The latent sample matrix.
        V: The latent trait matrix.
        neighbors: A python dictionary whose keys are integer indices corresponding to 
                   indices in range(g), and whose values are lists of indices corresponding
                   to neighbors of the keys.
        eta: The learning rate (multiplicative factor applied to the gradient).
        lamb: Hyper-parameter controlling how much to regularize the latent representations.
        num_epochs: The number of epochs to run SGD over. The default is 100.
        known_indices: An optional t x 2 matrix, each row of which represents the index
                       of a known entry in X. Used to train on only a subset of known entries. 
                       If a vector is provided, assumes that the vector denotes the indices of
                       samples to use as training.
                       If None is provided, then the algorithm will train over all non nan values.
                       
    Returns:
        Matrices U and V representing the latent vectors for each sample and each gene, respectively.
    '''
    if (known_indices is None):
        known_indices = np.argwhere(~np.isnan(X))
    
    Factor(X, U, V, known_indices, neighbors, eta, lamb1, lamb2, num_epochs)
    
    return U, V

def GetRepresentationError(X, U, V, known_indices):
    return Get_prediction_error(X, U, V, known_indices)
        