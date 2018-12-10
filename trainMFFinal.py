import numpy as np
import pandas as pd
import datetime

from multiprocessing import Pool
from functools import partial

from CrossValidation import *
from FeatureSimilarity import GetTopGenes
from MatrixFactorization import CreateLatentVariables, FactorizeMatrix, GetRepresentationError

from utils import *
from ReadData import *
from GetJSON import get

print('Reading in data...')
totalDataDF = pd.read_csv('/projects/leelab3/psturm/concatData/totalDataDF.csv', header=0, index_col=0)
binaryPathwayDF = pd.read_csv('/projects/leelab3/psturm/concatData/pathways.tsv', sep='\t', header=0)
binaryPathwayDF.set_index('Genes', inplace=True)

X = totalDataDF.values.T
n, g = X.shape

print('Projecting onto principal components...')
completeMat = totalDataDF.dropna(axis=0).values
pca = PCA(n_components=500)
projectedX = pca.fit_transform(completeMat.T)
latent_dim = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.90)[0])
print('Latent dimension is: {}'.format(latent_dim))

binaryPathwayMat = binaryPathwayDF.values
neighbors = GetNeighborDictionary(binaryPathwayMat)

eta   = 0.01
lamb1 = 0.04
lamb2 = 0.02

print('Factoring matrix...')
U_init, V_init = CreateLatentVariables(n, g, latent_dim)
U, V, trainError, testError = FactorizeMatrix(X, U_init, V_init, neighbors, eta=eta, lamb1=lamb1, lamb2=lamb2, num_epochs=10, returnErrorVectors=True)

np.save('/projects/leelab3/psturm/realModels/U.npy', U)
np.save('/projects/leelab3/psturm/realModels/V.npy', V)
np.save('/projects/leelab3/psturm/realModels/trainError.npy', trainError)
np.save('/projects/leelab3/psturm/realModels/testError.npy', testError)