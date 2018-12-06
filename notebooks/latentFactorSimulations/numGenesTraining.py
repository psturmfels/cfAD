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

numReps = 50

etas = [0.006872, 0.004405, 0.003169, 0.003933]
lambs_1 = [0.001322, 0.018094, 0.007227, 0.004865]
lambs_2 = [0.013549, 0.009637, 0.016451, 0.010235]
g_list = [1000, 3000, 5000, 7000]

def TrainReps(rep):
    print('-------------Training data on dataset {}-------------'.format(rep))
    for i in range(len(g_list)):
        g = g_list[i]
        eta = etas[i]
        lamb1 = lambs_1[i]
        lamb2 = lambs_2[i]
        
        print('rep {}, g {}'.format(rep, g))
        dataFileBase = '/projects/leelab3/psturm/simulatedData/varyDimData/g{}/df{}.csv'
        df = pd.read_csv(dataFileBase.format(g, rep))
        X, binaryPathwayMatrix, phenotypeGenes = DFtoDataset(df)
        n, _ = X.shape
        neighbors = GetNeighborDictionary(binaryPathwayMatrix, percentileThreshold=95)
        
        pca = PCA(n_components=50)
        projectedX = pca.fit_transform(X.T)
        latent_dim = np.min(np.where(np.cumsum(pca.explained_variance_ratio_) > 0.95)[0])

        U_pred_init, V_pred_init = CreateLatentVariables(n, g, latent_dim)
        U_pred, V_pred           = FactorizeMatrix(X, U_pred_init, V_pred_init, neighbors, 
                                         eta=eta, lamb1=lamb1, lamb2=lamb2, num_epochs=10)

        np.save('/projects/leelab3/psturm/simulatedModels/geneModels/g{}/U{}.npy'.format(g, rep), U_pred)
        np.save('/projects/leelab3/psturm/simulatedModels/geneModels/g{}/V{}.npy'.format(g, rep), V_pred)

numProcesses = 25
p = Pool(numProcesses)
p.map(TrainReps, range(numReps))
p.close()
p.join()