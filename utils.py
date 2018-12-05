import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.decomposition import PCA

def GenerateRegressedPhenotype(X, numPhenotypes=1, lam=1, binaryPathwayMatrix=None, coeffSigma=1.0):
    n, g = X.shape

    if binaryPathwayMatrix is not None:
        _, k = binaryPathwayMatrix.shape
    
    Y = np.zeros((n, numPhenotypes))
    geneCoeffs = np.zeros((g, numPhenotypes))
    for i in range(numPhenotypes):
        if binaryPathwayMatrix is not None:
            numPathways = np.minimum(np.random.poisson(lam=lam) + 1, k)
            chosenPathways = np.random.choice(k, size=(numPathways, ), replace=False)
            for l in chosenPathways:
                chosenIndices = np.where(binaryPathwayMatrix[:, l] > 0)[0]
                chosenIndices = np.unique(chosenIndices)
                numGenesInPhenotype = len(chosenIndices)
                geneCoeffs[chosenIndices, i] = np.random.choice([-1, 1]) * np.abs(np.random.randn(numGenesInPhenotype) * coeffSigma)
        else:
            numGenesInPhenotype = np.minimum(np.random.poisson(lam=lam) + 1, g)
            chosenIndices = np.random.choice(g, size=(numGenesInPhenotype,), replace=False)
            geneCoeffs[chosenIndices, i] = np.random.randn(numGenesInPhenotype) * coeffSigma
            
        Y[:, i] = np.dot(X[:, chosenIndices], geneCoeffs[chosenIndices, i]) + np.random.randn(n) * coeffSigma * 0.5
    
    return Y, geneCoeffs
            
    
#LATENT FACTOR MODEL GENERATION
def GenerateSimulatedData(n = 200, g = 2000, k = 20, avgGenesInPath=100.0, covariateU=False):
    sigma = 0.5
    binaryPathwayMatrix = np.zeros((g, k))
    remainingGeneIndices = np.arange(1, g)
    
    if covariateU:
        randomMat = np.random.randn(k, k).astype(np.float) * sigma;
        covMat = np.dot(randomMat.T, randomMat)
        covMat = covMat / np.max(covMat)
        covMat = covMat + np.maximum(0.5 - np.mean(np.diag(covMat)), 0.0) * np.eye(k)
        mean = np.zeros((k,))
        U = np.random.multivariate_normal(mean, covMat, size=(n,))
    else:
        U = np.random.randn(n, k).astype(np.float32) * sigma;
    
    V = np.random.randn(g, k).astype(np.float32) * sigma;
    
    pathwaySizes = np.random.poisson(lam=avgGenesInPath, size=(k, )) + 1
    pathwaySizes = (pathwaySizes / np.sum(pathwaySizes)) * g
    pathwaySizes = pathwaySizes.astype(int)
    pathwaySizes[0] += g - 1 - np.sum(pathwaySizes)
    
    for i in range(k):
        numIndices = np.maximum(np.random.randint(low=int(k/4), high=k+1), 1)
        means = np.random.randn(numIndices).astype(np.float32) * sigma
        sigmas = np.random.uniform(low=0.0, high=sigma, size=(numIndices))
        chosenIndices = np.random.choice(k, size=(numIndices,), replace=False)
        
        numGenes = pathwaySizes[i]
        chosenGeneIndices = np.random.choice(len(remainingGeneIndices), size=(numGenes,), replace=False)
        chosenGenes = remainingGeneIndices[chosenGeneIndices]
        
        remainingGeneIndices = np.delete(remainingGeneIndices, chosenGeneIndices)
        
        if i == 0:
            chosenGenes = np.append(chosenGenes, 0)
            phenotypeGenes = chosenGenes
            numGenes = numGenes + 1
        
        V[chosenGenes[:, None], chosenIndices] = np.random.multivariate_normal(means, 
                                                                                  np.diag(sigmas),
                                                                                  size=(numGenes,))
        binaryPathwayMatrix[chosenGenes, i] = 1
        
    
    binaryPathwayMatrix[0, :] = np.zeros(k)
    return U, V, binaryPathwayMatrix, phenotypeGenes

#Helper functions
def GetNeighborDictionary(binaryPathwayMatrix, percentileThreshold=95):
    neighbors = {}

    nonzeroIndices = np.where(np.any(binaryPathwayMatrix, axis=1))[0]
    nonzeroIndices = nonzeroIndices.astype(np.int32)
    nonzeroPathwayMat = binaryPathwayMatrix[nonzeroIndices, :]
    numNonzero, k = nonzeroPathwayMat.shape

    geneDegreeMatrix  = np.dot(nonzeroPathwayMat, nonzeroPathwayMat.T)
    np.fill_diagonal(geneDegreeMatrix, 0.0)
    
    degreePercentiles = np.percentile(geneDegreeMatrix, percentileThreshold, axis=1)
    geneDegreeMatrix[geneDegreeMatrix < np.expand_dims(degreePercentiles, axis=1)] = 0
    
    geneDegreeCounts  = np.sum(geneDegreeMatrix, axis=1)

    for i in range(numNonzero):
        geneDegree = geneDegreeCounts[i]
        if (geneDegree == 0):
            continue

        neighbors[nonzeroIndices[i]] = []
        neighborEdgeIndices = geneDegreeMatrix[i, :].nonzero()[0]
        neighborEdgeWeights = geneDegreeMatrix[i, neighborEdgeIndices]
        for j in range(len(neighborEdgeIndices)):
            neighbors[nonzeroIndices[i]].append([
                nonzeroIndices[neighborEdgeIndices[j]],
                neighborEdgeWeights[j] / geneDegree
            ])
    
    return neighbors

def MatToMeltDF(im, group_name, x_values=np.arange(400)):
    numReps, numPlotPoints = im.shape
    if len(x_values) > numPlotPoints:
        im = np.concatenate([im, np.tile(im[:, -1], (len(x_values) - numPlotPoints, 1)).T], axis=1)
    
    im_dot_df = pd.DataFrame(im[:, :len(x_values)].T)
    im_dot_df['percent identified as significant'] = x_values
    im_dot_df = pd.melt(im_dot_df, id_vars=['percent identified as significant'], 
                        value_name='percent identified actually significant')
    im_dot_df['group'] = group_name
    return im_dot_df

def GetMeanErrorDF(errorsDF, num_folds=5):
    meanErrorsDF = pd.concat([errorsDF[errorsDF['fold'] == i].drop('fold', axis=1).reset_index(drop=True) for i in range(num_folds)], axis=0)
    meanErrorsDF = meanErrorsDF.groupby(meanErrorsDF.index).mean()
    return meanErrorsDF

def ScreePlot(X, var_ratio=0.9):
    X = X - np.mean(X, axis=0)
    pca_model = PCA()
    pca_model.fit(X)
    
    latent_dim = np.min(np.where(np.cumsum(pca_model.explained_variance_ratio_) > var_ratio)[0])
    plt.axvline(latent_dim, color='orange')
    
    ax = plt.gca()
    ax2 = plt.twinx()
    exp_var = sns.lineplot(x=np.arange(len(pca_model.explained_variance_ratio_)), y=pca_model.explained_variance_ratio_, ax=ax, color='b', label='Explained variance')
    sum_var = sns.lineplot(x=np.arange(len(pca_model.explained_variance_ratio_)), y=np.cumsum(pca_model.explained_variance_ratio_), ax=ax2, color='r', label='Cumulative explained variance')
    
    plt.title('Scree Plot with latent dimension {}'.format(latent_dim))
    
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc=2)
    ax2.get_legend().remove()
        
    ylim1 = ax.get_ylim()
    len1 = ylim1[1]-ylim1[0]
    yticks1 = ax.get_yticks()
    rel_dist = [(y-ylim1[0])/len1 for y in yticks1]
    ylim2 = ax2.get_ylim()
    len2 = ylim2[1]-ylim2[0]
    yticks2 = [ry*len2+ylim2[0] for ry in rel_dist]

    ax2.set_yticks(yticks2)
    ax2.set_ylim(ylim2)
    ax.set_xlabel('Principal components')
    ax.set_ylabel('Percent variance')
    ax2.set_ylabel('Cumuluative percent variance')
    
    ax.set_axisbelow(True)
    ax2.set_axisbelow(True)
    ax2.grid(False)