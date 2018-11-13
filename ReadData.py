import numpy as np
import pandas as pd

def GetDataFrame(filename, sep='\t', header=0):
    '''
    Parses the data frame stored in filename.
    
    Args:
        filename: The file name of the data frame to read.
        sep: The separating set of characters between each entry in the file.
        header: An integer representing the index the header is stored in.
        
    Returns:
        A data frame read from filename.
    '''
    return pd.read_csv(filename, sep=sep, header=header)

def JoinGenes(df1, df2, gene_col_name='PCG'):
    '''
    Joins two data frames that represent gene expression matrices.
    Each row represents a gene, and each column represents a sample.
    
    Args:
        df1: The first data frame.
        df2: The second data frame. 
        gene_col_name: The name of the column that contains the gene names.

    Returns: A data frame that represents merging the two input data frames on
             the gene column name.

    '''
    return df1.merge(df2, on=gene_col_name)

def JoinGenePheno(geneDF, phenoDF):
    '''
    Joins a gene expression data frame and a phenotype data frame.
    
    Args:
        geneDF: A data frame representing a gene expression matrix.
        phenoDF: A data frame representing a phenotype matrix.
        
    Returns: A matrix that represents stacking the two data frames on top of each other,
             e.g., a new data frame where each row represents a biological trait (gene
             expression or phenotype), and each column represents a sample.
    '''
    phenoDF = phenoDF.set_index('sample_name').T.rename_axis('PCG').rename_axis(None, 1).reset_index()
    return pd.concat([geneDF, phenoDF], ignore_index=True, sort=False)

def JoinMultipleGenes(*dfs):
    '''
    Wrapper function to combine multiple gene expression data frames
    horizontally.
    
    Args:
        *dfs: An unwrapped list of gene expression data frames. 
    
    Returns: A data frame in which all of the input data frames has been combined.
    
    Raises: 
        ValueError: Raised if no inputs are given. 
    ''' 
    if len(dfs) == 0:
        raise ValueError('Cannot join an empty list of gene data frames.')
    base_df = dfs[0]
    for df in dfs[1:]:
        base_df = JoinGenes(base_df, df)
    return base_df