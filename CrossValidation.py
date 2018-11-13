import numpy as np
from sklearn.model_selection import KFold
from MatrixFactorization import FactorizeMatrix, GetRepresentationError
from FeatureSimilarity import GetTopGenes


def CrossValidation(X, k, etas, lambs, foldcount=5):
    '''
    Runs the matrix factorization algorithm for each specified value of eta and lambda
    and computes the reconstruction errors for each run.

    Args:
        X: An n x g, possibly sparse numpy matrix, where missing entries are indicated by np.nan values,
           where n represents the number of samples and g represents the number of genes, or items.
        k: The latent dimension of the factorization. Typically, k < min(n, g).
        etas: A list or vector of values containing learning rates.
        lambs: A list or vector of values containing regularization strengths.\
        foldcount: An integer denoting the number of folds for cross validation.
        
    Returns:
        A len(etas) x len(lambs) x foldcount tensor denoting the reconstruction error for each
        setting of eta and lambda on each fold.
    '''
    kf = KFold(n_splits=foldcount, shuffle=True)
    errors = np.zeros((len(etas), len(lambs), foldcount))
    
    fold = 0
    for train_index, test_index in kf.split(X):
        j = 0
        for lamb in lambs:
            i = 0
            for eta in etas:
                U, V = FactorizeMatrix(X, k, eta=eta, lamb=lamb, known_indices=train_index)
                errors[i, j, fold] = GetRepresentationError(X, U, V, known_indices=test_index)
                i = i + 1
            j = j + 1
        fold = fold + 1
    return errors

def GetGeneOverlap(X, k, num_trials=100):
    