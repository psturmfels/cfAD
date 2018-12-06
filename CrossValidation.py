import numpy as np
import pandas as pd
import seaborn as sns; sns.set()
import matplotlib.pyplot as plt

from multiprocessing import Pool
from functools import partial

from sklearn.model_selection import KFold
from MatrixFactorization import FactorizeMatrix, GetRepresentationError, CreateLatentVariables
from FeatureSimilarity import GetTopGenes

def RandomParams(eta_low, eta_high, lamb1_low, lamb1_high, lamb2_low, lamb2_high, num_reps=20):
    hyper_params = np.zeros((num_reps, 3)).astype(np.float32)
    hyper_params[:, 0] = np.random.uniform(low=eta_low, high=eta_high, size=(num_reps,))
    hyper_params[:, 1] = np.random.uniform(low=lamb1_low, high=lamb1_high, size=(num_reps,))
    hyper_params[:, 2] = np.random.uniform(low=lamb2_low, high=lamb2_high, size=(num_reps,))
    return hyper_params
    
def TrainOnParams(params, X, k, neighbors, train_indices, test_indices):
    print('.', end='')
    n, g = X.shape
    eta, lamb1, lamb2 = params
    U, V = CreateLatentVariables(n, g, k)
    U, V = FactorizeMatrix(X, U, V, neighbors, eta=eta, lamb1=lamb1, lamb2=lamb2, trainIndices=train_indices)
    paramError = GetRepresentationError(X, U, V, known_indices=test_indices)
    return paramError

def TrainVerboseOnParams(params, X, k, neighbors, train_indices, test_indices):
    print('.', end='')
    n, g = X.shape
    eta, lamb1, lamb2 = params
    U, V = CreateLatentVariables(n, g, k)
    U, V, trainError, testError = FactorizeMatrix(X, U, V, neighbors, eta=eta, lamb1=lamb1, lamb2=lamb2, trainIndices=train_indices, returnErrorVectors=True)
    paramError = GetRepresentationError(X, U, V, known_indices=test_indices)
    return paramError, trainError, testError

def CrossValidation(X, k, hyper_params, neighbors=None, foldcount=5, returnVectorDF=False, numProcesses=20):
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
    errorsDF = pd.DataFrame(np.zeros((len(hyper_params) * foldcount, 5)))
    errorsDF.columns = ['eta', 'lamb1', 'lamb2', 'error', 'fold']
    
    #Okay not to shuffle because kf shuffles for you
    known_indices = np.argwhere(~np.isnan(X)).astype(np.int32)
    np.random.shuffle(known_indices)
    
    if returnVectorDF:
        trainErrorDF = pd.DataFrame()
        testErrorDF  = pd.DataFrame()
    
    fold = 0
    df_index = 0
    
    p = Pool(numProcesses)
    
    for train_index, test_index in kf.split(known_indices):
        print('Training fold {}'.format(fold))
        if returnVectorDF:
            foldTrainDF = pd.DataFrame()
            foldTestDF  = pd.DataFrame()
        
        train_indices = known_indices[train_index].astype(np.int32)
        test_indices  = known_indices[test_index].astype(np.int32)
        
        if (returnVectorDF):
            errorVec = p.map(partial(TrainVerboseOnParams, X=X, k=k, neighbors=neighbors,
                                     train_indices=train_indices, test_indices=test_indices), hyper_params)
            for i in range(len(hyper_params)):
                eta, lamb1, lamb2 = hyper_params[i]
                paramError, trainError, testError = errorVec[i]
                foldTrainDF = pd.concat([foldTrainDF,
                                      pd.DataFrame({
                                          'eta{:.5f}_lamb1{:.5f}_lamb2{:.5f}'.format(eta, lamb1, lamb2): trainError
                                      })
                                    ], axis=1)
                foldTestDF  = pd.concat([foldTestDF,
                                          pd.DataFrame({
                                          'eta{:.5f}_lamb1{:.5f}_lamb2{:.5f}'.format(eta, lamb1, lamb2): testError
                                      })
                                    ], axis=1)
                errorsDF.iloc[df_index] = np.array([eta, lamb1, lamb2, paramError, fold])
                df_index += 1
        else:
            errorVec = p.map(partial(TrainOnParams, X=X, k=k, neighbors=neighbors,
                                     train_indices=train_indices, test_indices=test_indices), hyper_params)
            for i in range(len(hyper_params)):
                eta, lamb1, lamb2 = hyper_params[i]
                paramError = errorVec[i]
                errorsDF.iloc[df_index] = np.array([eta, lamb1, lamb2, paramError, fold])
                df_index += 1
        
        if returnVectorDF:
            foldTrainDF['fold'] = fold
            foldTestDF['fold']  = fold
            maxEpochs, _ = foldTrainDF.shape
            foldTrainDF['epochs']  = np.arange(maxEpochs).astype(np.float32)
            foldTestDF['epochs']   = np.arange(maxEpochs).astype(np.float32)
            
            trainErrorDF = pd.concat([trainErrorDF, foldTrainDF])
            testErrorDF  = pd.concat([testErrorDF, foldTestDF])
            
        fold = fold + 1
    
    p.close()
    p.join()
    
    if returnVectorDF:
        return errorsDF, trainErrorDF, testErrorDF 
    else:
        return errorsDF

def PlotErrorDF(errorDF, id_vars=['epochs', 'fold'], ax=None):
    data = pd.melt(errorDF, id_vars=id_vars, value_name='error', var_name='run')
    if ax is not None:
        ax = sns.lineplot(x='epochs', y ='error', hue='run', data=data, ax=ax, legend=False)
    else: 
        ax = sns.lineplot(x='epochs', y ='error', hue='run', data=data, legend='brief')
        plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return ax

def PlotParamDF(paramDF, id_vars=['error', 'fold'], ax=None):
    data = pd.melt(paramDF, id_vars=id_vars, value_name='param_value', var_name='param_type')
    if ax is not None:
        ax = sns.lineplot(x='param_value', y='error', hue='param_type', data=data, ax=ax)
    else:
        ax = sns.lineplot(x='param_value', y='error', hue='param_type', data=data)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    return ax