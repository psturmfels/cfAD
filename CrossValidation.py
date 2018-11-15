import numpy as np
from sklearn.model_selection import KFold
from MatrixFactorization import FactorizeMatrix, GetRepresentationError
from FeatureSimilarity import GetTopGenes

def CrossValidation(X, k, hyper_params, foldcount=5):
    '''
    Runs the matrix factorization algorithm for each specified value of eta and lambda
    and computes the reconstruction errors for each run.

    Args:
        X: An n x g, possibly sparse numpy matrix, where missing entries are indicated by np.nan values,
           where n represents the number of samples and g represents the number of genes, or items.
        k: The latent dimension of the factorization. Typically, k < min(n, g).
        hyper_params: A list of tuples, each corresponding to a setting of hyper parameters (eta, lamb1, lamb2).
        foldcount: An integer denoting the number of folds for cross validation.
        
    Returns:
        A len(etas) x len(lambs) x foldcount tensor denoting the reconstruction error for each
        setting of eta and lambda on each fold.
    '''
    n, g = X.shape
    
    kf = KFold(n_splits=foldcount, shuffle=True)
    errors = np.zeros((len(hyper_params), foldcount))
    
    #Okay not to shuffle because kf shuffles for you
    known_indices = np.argwhere(~np.isnan(X)).astype(np.int32)
    
    fold = 0
    for train_index, test_index in kf.split(known_indices):
        i = 0
        for eta, lamb1, lamb2 in hyper_params:
            train_indices = known_indices[train_index]
            test_indices  = known_indices[test_index]

            U, V = CreateLatentVariables(n, g, k)
            U, V = FactorizeMatrix(X, U, V, neighbors, eta=eta, lamb1=lamb1, lamb2=lamb2, trainIndices=train_indices)
            errors[i, fold] = GetRepresentationError(X, U, V, known_indices=test_indices)
                  
            i = i + 1
        fold = fold + 1
    return errors
    