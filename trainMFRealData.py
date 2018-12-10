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

totalDataDF = pd.read_csv('/projects/leelab3/psturm/concatData/totalDataDF.csv', header=0, index_col=0)
binaryPathwayDF = pd.read_csv('/projects/leelab3/psturm/concatData/pathways.tsv', sep='\t', header=0)
binaryPathwayDF.set_index('Genes', inplace=True)

X = totalDataDF.values.T

n, g = X.shape
half_n = int(n / 2)

binaryPathwayMat = binaryPathwayDF.values
neighbors = GetNeighborDictionary(binaryPathwayMat)

eta   = 0.01
lamb1 = 0.04
lamb2 = 0.02
# eta_nn   = 
# lamb1_nn = 
# lamb2_nn = 

latentDim = 100 #Somewhat arbitrary, but solution does not vary greatly as a function of latent dimension
numReps   = 100 #Also somewhat arbitrary. Lower this if it takes too long.
maxEpochs = 4   #Based on CV results. Since the data matrix is so large, it doesn't take many epochs to converge

def TrainReps(rep):
    with open('trainMF_real.txt', 'a') as progress_file:
        progress_file.write('Started random split {} at time:\t{}\n'.format(rep, datetime.datetime.now()))
    
    randomIndices = np.loadtxt('/projects/leelab3/psturm/realData/randomIndices/perm{}.csv'.format(rep), dtype=int)
    randomIndices = randomIndices[randomIndices < n]
    trainIndices  = randomIndices[:half_n]
    valdIndices   = randomIndices[half_n:]
    
    trainX = X[trainIndices, :]
    valdX  = X[valdIndices, :]
    
    U_init_train, V_init_train = CreateLatentVariables(len(trainIndices), g, latentDim)
    U_train, V_train = FactorizeMatrix(trainX, U_init_train, V_init_train, neighbors, 
                                          eta=eta, lamb1=lamb1, lamb2=lamb2, num_epochs=maxEpochs)
    
    U_init_vald, V_init_vald = CreateLatentVariables(len(valdIndices), g, latentDim)
    U_vald, V_vald = FactorizeMatrix(valdX, U_init_vald, V_init_vald, neighbors, 
                                          eta=eta, lamb1=lamb1, lamb2=lamb2, num_epochs=maxEpochs)
    
    np.save('/projects/leelab3/psturm/realModels/overlapModels/U_train{}.npy'.format(rep), U_train)
    np.save('/projects/leelab3/psturm/realModels/overlapModels/V_train{}.npy'.format(rep), V_train)
    np.save('/projects/leelab3/psturm/realModels/overlapModels/U_vald{}.npy'.format(rep), U_vald)
    np.save('/projects/leelab3/psturm/realModels/overlapModels/V_vald{}.npy'.format(rep), V_vald)
    
    with open('trainMF_real.txt', 'a') as progress_file:
        progress_file.write('Ended random split {} at time:\t{}\n'.format(rep, datetime.datetime.now()))

numProcesses = 25
p = Pool(numProcesses)
p.map(TrainReps, range(numReps))
p.close()
p.join()