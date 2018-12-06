import sys
import os
import datetime

module_path = os.path.abspath(os.path.join('../..'))
if module_path not in sys.path:
    sys.path.append(module_path)

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from multiprocessing import Pool
from functools import partial

from CrossValidation import *
from FeatureSimilarity import GetTopGenes
from MatrixFactorization import CreateLatentVariables, FactorizeMatrix, GetRepresentationError

from utils import *

def DFtoDataset(df, scale=False):
    n = 500
    X = df[[str(i) for i in np.arange(n)]].values.T
    if (scale):
        X = preprocessing.scale(X)
        
    binaryPathwayMatrix = df[['pathway{}'.format(i) for i in range(df.shape[1] - n - 2)]].values

    phenotypeGenes = df['phenotype_genes']
    phenotypeGenes = np.where(phenotypeGenes == 1)[0]

    return X, binaryPathwayMatrix, phenotypeGenes

for g in [1000, 3000, 5000, 7000]:
    print('-------------Tuning on data with {} genes-------------'.format(g))
    dataFileBase = '/projects/leelab3/psturm/simulatedData/varyDimData/g{}/df{}.csv'
    df = pd.read_csv(dataFileBase.format(g, 0))
    X, binaryPathwayMatrix, phenotypeGenes = DFtoDataset(df)
    neighbors=GetNeighborDictionary(binaryPathwayMatrix)
    
    pca = PCA(n_components=50)
    pca.fit(X.T)
    latent_dim = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0])
    
    num_folds=5
    hyper_params = RandomParams(eta_low=0.001, eta_high=0.02, lamb1_low=0.001, lamb1_high=0.04, lamb2_low=0.001, lamb2_high=0.02, num_reps=50)
    errorsDF, trainErrorDF, testErrorDF = CrossValidation(X, latent_dim, hyper_params, neighbors=neighbors, foldcount=num_folds, returnVectorDF=True, numProcesses=25)
    
    errorsDF.to_csv('../../DataFrames/errorsDF_g{}.csv'.format(g), index=False)
    trainErrorDF.to_csv('../../DataFrames/trainErrorDF_g{}.csv'.format(g), index=False)
    testErrorDF.to_csv('../../DataFrames/testErrorDF_g{}.csv'.format(g), index=False)