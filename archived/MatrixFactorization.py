import numpy as np

def FactorizeMatrix(X, k, eta=0.005, lamb=0.02, num_epochs=200, known_indices=None, test_indices=None, verbose=False):
    '''
    Factorizes the sparse matrix X into the product of two rank k matrices
    U and V using stochastic gradient descent.
    
    Args:
        X: An n x g, possibly sparse numpy matrix, where missing entries are indicated by np.nan values,
           where n represents the number of samples and g represents the number of genes, or items.
        k: The latent dimension of the factorization. Typically, k < min(n, g).
        eta: The learning rate (multiplicative factor applied to the gradient).
        lamb: Hyper-parameter controlling how much to regularize the latent representations.
        num_epochs: The number of epochs to run SGD over. The default is 100.
        known_indices: An optional t x 2 matrix, each row of which represents the index
                       of a known entry in X. Used to train on only a subset of known entries. 
                       If a vector is provided, assumes that the vector denotes the indices of
                       samples to use as training.
                       If None is provided, then the algorithm will train over all non nan values.
                       
        verbose: Whether or not to print out the current epoch while training. Defaults to False.
                       
    Returns:
        Matrices U and V representing the latent vectors for each sample and each gene, respectively.
    '''
    n, g = X.shape
    
    #The shape allows us to interpret the rows of U and V as latent representations.
    sigma = 0.02
    U = np.random.randn(n, k) * sigma
    V = np.random.randn(g, k) * sigma
    
    if (known_indices is None):
        known_indices = np.argwhere(~np.isnan(X))
    
    for epoch in range(num_epochs):
        np.random.shuffle(known_indices)
        
        if len(known_indices.shape) == 2:
            iterated_indices = known_indices
        elif len(known_indices.shape) == 1:
            iterated_indices = np.array(np.meshgrid(known_indices, np.arange(g))).T.reshape(-1, 2)
        else:
            raise ValueError('known_indices has shape {}, but should be 1D or 2D.'.format(known_indices.shape))
        
        for known_index in iterated_indices:
            i, j = known_index

            x_ij = X[i, j]
            u_i  = U[i, :]
            v_j  = V[j, :]

            #Calculate symbolic gradients
            e_ij = x_ij - np.dot(u_i, v_j)
            grad_ui = e_ij * v_j - lamb * u_i
            grad_vj = e_ij * u_i - lamb * v_j

            #Apply gradients to latent representation
            U[i, :] = u_i + eta * grad_ui
            V[j, :] = v_j + eta * grad_vj
        
        if (verbose and epoch % 1 == 0):
            train_error = GetRepresentationError(X, U, V, known_indices)
            test_error  = GetRepresentationError(X, U, V, test_indices)
            print('Epoch {} - current train error: {} - current test error: {}'.format(epoch, train_error, test_error))
    
    return U, V

def GetRepresentationError(X, U, V, known_indices=None):
    '''
    Calculates the mean reconstruction error between x_ij and u_i^T v_j.
    
    Args:
        X: An n x g, possibly sparse numpy matrix, where missing entries are indicated by np.nan values,
           where n represents the number of samples and g represents the number of genes, or items.
        U: An n x k matrix whose rows represent the latent sample vectors.
        V: An n x g matrix whose rows represent the latent gene vectors.
        known_indices: An optional t x 2 matrix, each row of which represents the index
                       of a known entry in X. Used to train on only a subset of known entries. 
                       If a vector is provided, assumes that the vector denotes the indices of
                       samples to use as training.
                       If None is provided, then the algorithm will train over all non nan values.
    
    Returns:
        Mean reconstruction error of UV^T in estimating X.
    '''
    n, g = X.shape
    
    if (known_indices is None):
        known_indices = np.argwhere(~np.isnan(X))
    
    error = 0
    if len(known_indices.shape) == 2:
        iterated_indices = known_indices
    elif len(known_indices.shape) == 1:
        iterated_indices = np.array(np.meshgrid(known_indices, np.arange(g))).T.reshape(-1, 2)
    else:
        raise ValueError('known_indices has shape {}, but should be 1D or 2D.'.format(known_indices.shape))
    
    num_known, _ = iterated_indices.shape
    for known_index in iterated_indices:
        i, j = known_index
        
        x_ij = X[i, j]
        u_i  = U[i, :]
        v_j  = V[j, :]
        
        error = error + np.square(x_ij - np.dot(u_i, v_j))
    error = error / num_known
    return error

n = 1000
g = 20000
latentDim = 50

#Create some random, low-rank data
U = np.random.randn(n, latentDim).astype(np.float32)
V = np.random.randn(g, latentDim).astype(np.float32)
X = np.dot(U, V.T)

knownIndices = np.argwhere(~np.isnan(X))
#For testing purposes, we need to shuffle the indices. If we do not,
#we will be training on a chunk of the upper half of the matrix, but
#testing on the lower half of the matrix. This makes no sense,
#because we don't have any information about the latent variables we are testing on. Therefore, shuffle!
np.random.shuffle(knownIndices)

numberTestIndices = 20000
    
testIndices  = knownIndices[:numberTestIndices, :]
trainIndices = knownIndices[numberTestIndices:, :]
print(testIndices)
print(trainIndices)

U, V = FactorizeMatrix(X, k=latentDim, eta=0.005, lamb=0.02, num_epochs=5, known_indices=trainIndices, test_indices=testIndices, verbose=True)
testError = GetRepresentationError(X, U, V, known_indices=testIndices)
print('Final Test Error was: {}'.format(testError))
