import numpy as np

def GetTopGenes(V, phenotype_index, gene_indices=None, c=None, rankType='dot', sortFunc=np.abs):
    '''
    Returns the top genes associated with phenotype given by pheno_type index, assuming
    V is a latent gene-phenotype matrix.
    
    Args:
        V: A g x k matrix where each row represents the latent representation of a gene or phenotype.
        phenotype_index: An index in [0, g - 1] that represents the target phenotype.
        gene_indices: An optional parameter denoting which rows of V to search through for top genes.
                      If None, searches through all rows of V.
        c: An optional parameter denoting how many genes to return. If c is None,
           returns all genes.
    
    Returns:
        A list of indices corresponding to the rows of V, sorted in order of relevance
        to the target phenotype.
    '''
    #NOTE: Try changing this to cosine similarity? Or correlation?
    phenotype_vector = V[phenotype_index, :]
    
    if gene_indices is not None:
        V = V[gene_indices, :]

    assert rankType in ['dist', 'corr', 'cos', 'dot'], 'rankType must be one of dist, corr, cos, dot'
    if rankType == 'dist':
        association_scores = -1.0 * np.linalg.norm(V - phenotype_vector, axis=1)
    else:
        if rankType == 'corr':
            phenotype_vector = phenotype_vector - np.nanmean(phenotype_vector)
            V = V - np.nanmean(V, axis=1, keepdims=True)

        association_scores = np.dot(V, phenotype_vector)
        if rankType == 'cos' or rankType == 'corr':
            pheno_norm = np.linalg.norm(phenotype_vector)
            V_norms    = np.linalg.norm(V, axis=1)
            association_scores = association_scores / (pheno_norm * V_norms)
    
    if sortFunc is not None:
        top_gene_indices = sortFunc(association_scores).argsort()[::-1]
    else:
        top_gene_indices = association_scores.argsort()[::-1]
    
    if gene_indices is not None:
        top_gene_indices = gene_indices[top_gene_indices]
        
    if c is not None:
        top_gene_indices = top_gene_indices[:c]
        
    return top_gene_indices

def GetTopGenesMulti(V, phenotype_indices, gene_indices=None, aggFunc=np.mean):
    phenotype_vector = V[phenotype_indices, :].T
    if gene_indices is not None:
        V = V[gene_indices, :]
    association_scores = -1.0 * aggFunc(np.linalg.norm(V[:, :, None] - phenotype_vector[None, :, :], axis=1), axis=1)
    top_gene_indices = association_scores.argsort()[::-1]
    if gene_indices is not None:
        top_gene_indices = gene_indices[top_gene_indices]
    return top_gene_indices


