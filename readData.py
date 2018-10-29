import numpy as np
import pandas as pd
from getJSON import *

def GetDataFrame(filename, sep='\t', header=0):
    return pd.read_csv(filename, sep=sep, header=header)

def JoinGenes(df1, df2, gene_col_name='PCG'):
    return df1.merge(df2, on=gene_col_name)

def JoinGenePheno(geneDF, phenoDF):
    phenoDF = phenoDF.set_index('sample_name').T.rename_axis('PCG').rename_axis(None, 1).reset_index()
    return pd.concat([geneDF, phenoDF], ignore_index=True, sort=False)

def JoinMultipleGenes(*dfs):
    if len(dfs) == 0:
        return
    base_df = dfs[0]
    for df in dfs[1:]:
        base_df = JoinGenes(base_df, df)
    return base_df