import numpy as np

def GenerateSimulatedData(n = 200, g = 2000, k = 20, avgGenesInPath=100.0, covariateU=False, covariateV=False):
    sigma = 0.5 
    numPhenotypePathways = np.maximum(np.random.randint(low=int(k/4), high=k+1), 1)
    binaryPathwayMatrix = np.zeros((g, k + numPhenotypePathways)).astype(np.int32)
    pathwayMeans  = np.random.randn(k).astype(np.float32) * sigma
    pathwaySigmas = np.random.uniform(low=0.0, high=sigma, size=(k))
    
    if covariateU:
        randomMat = np.random.randn(k, k).astype(np.float) * sigma;
        covMat = np.dot(randomMat.T, randomMat)
        covMat = covMat / np.max(covMat)
        covMat = covMat + np.maximum(0.5 - np.mean(np.diag(covMat)), 0.0)
        mean = np.zeros((k,))
        U = np.random.multivariate_normal(mean, covMat, size=(n,))
    else:
        U = np.random.randn(n, k).astype(np.float32) * sigma;
    if covariateV:
        #FIGURE OUT HOW TO CREATE A RANDOM BLOCK COVARIANCE MATRIX 
        #BLOCKS WOULD REPRESENT PATHWAYS?
        randomMat = np.random.randn(k, k).astype(np.float) * sigma;
        covMat = np.dot(randomMat.T, randomMat)
        covMat = covMat / np.max(covMat)
        covMat = covMat + np.maximum(0.5 - np.mean(np.diag(covMat)), 0.0)
        mean = np.zeros((k,))
        V = np.random.multivariate_normal(mean, covMat, size=(g,))
    else:
        V = np.random.randn(g, k).astype(np.float32) * sigma;
    
    #Genes
    genesInPathCap = int(g/2) #The maximum number of genes in any pathway
    phenotypeGeneCap = np.maximum(int(genesInPathCap/10), 4)
    genesInPathIndicesCutoff = np.random.choice(np.arange(1, g), size=(genesInPathCap,), replace=False).astype(np.int32) 
    
    phenotypeGenes = genesInPathIndicesCutoff[:phenotypeGeneCap] #The genes that share the same pathways as the phenotype
    phenotypeGenes = np.append(phenotypeGenes, 0) #This index will represent the phenotype
    genesInPathIndicesCutoff = genesInPathIndicesCutoff[phenotypeGeneCap:] #The other genes belonging to some pathway
    
    numGenesInRandomPaths = len(genesInPathIndicesCutoff)
    numGenesInPhenoPath   = len(phenotypeGenes)
    
    for path in range(k):
        pathMean  = pathwayMeans[path]
        pathSigma = pathwaySigmas[path]
        numGenes  = np.minimum(np.random.geometric(p=1.0/avgGenesInPath) + 1, numGenesInRandomPaths)
        selectedGeneIndices = np.random.choice(genesInPathIndicesCutoff, size=(numGenes,), replace=False)
        numSelected = len(selectedGeneIndices)
        
        V[selectedGeneIndices, path] = np.random.randn(numSelected) * pathSigma + pathMean
        
        binaryPathwayMatrix[selectedGeneIndices, path] = 1
        
    phenotypePathways    = np.random.choice(k, size=(numPhenotypePathways), replace=False)
    phenotypeMeans  = np.random.randn(numPhenotypePathways).astype(np.float32) * sigma
    phenotypeSigmas = np.random.uniform(low=0.0, high=sigma, size=(numPhenotypePathways))
    
    binaryPathwayMatrix[phenotypeGenes[:, None], np.arange(numPhenotypePathways) + k] = 1
    binaryPathwayMatrix[0, :] = 0
    V[phenotypeGenes[:, None], phenotypePathways] = np.random.multivariate_normal(phenotypeMeans, 
                                                                                  np.diag(phenotypeSigmas),
                                                                                  size=(numGenesInPhenoPath,))
    
    
    return U, V, binaryPathwayMatrix, phenotypeGenes

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