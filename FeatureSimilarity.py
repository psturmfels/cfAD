import numpy as np

def GetTopGenes(V, phenotype_index, gene_indices=None, c=None):
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
    phenotype_vector = V[phenotype_index, :]
    
    if gene_indices is not None:
        V = V[gene_indices, :]
    
    association_scores = np.dot(V, phenotype_vector)
    top_gene_indices = association_scores.argsort()[::-1]
    
    if gene_indices is not None:
        top_gene_indices = gene_indices[top_gene_indices]
        
    if c is not None:
        top_gene_indices = top_gene_indices[:c]
        
    return top_gene_indices