library('SGL')
library('grpregOverlap')
library('MASS')
library('glmnet')
library('parallel')

numPheno <- 1
numGenes <- 2999
numSamples <- 500
numPaths <- 20

readDF <- function(i, dataFileBase='/projects/leelab3/psturm/simulatedData/latentData/df%i.csv') {
    ret <- list()
    
    df <- read.csv(sprintf(dataFileBase, i), header=TRUE)
    df = as.data.frame(df)
    df_t = as.data.frame(t(df))
    colnames(df_t) = c(paste('p', sep='', 1:numPheno), paste('g', sep='', 1:numGenes))
    data_mat = df_t[1:numSamples, ]

    phenotype_genes <- which(df[['phenotype_genes']][-1] != 0)
    y <- data_mat$p1
    x <- data.matrix(data_mat[, paste('g', sep='', 1:numGenes)])

    bin_path_mat <- df[paste('pathway', 0:(numPaths-1), sep='')]
    bin_path_mat <- bin_path_mat[-1, ]

    group_index <- integer(numGenes) - 1
    group_obj <- which(bin_path_mat == 1, arr.ind=T)
    group_index[as.vector(group_obj[, 1])] <- as.vector(group_obj[, 2])
    
    ret$x <- x
    ret$y <- y
    ret$group_index <- group_index
    
    ret$phenotype_genes <- phenotype_genes
    
    ret$group_df <- lapply(unique(group_index), function(o, gi) { 
                        paste('g', which(gi == o), sep='')
                        }, 
                group_index)
    return(ret)
}

getCounts <- function(coeffs, reference_index) {
    discovered_genes <- which(coeffs != 0)
    discovered_genes <- discovered_genes[order(abs(coeffs[discovered_genes]), decreasing=TRUE)]
    num_discovered <- length(discovered_genes)
    total_count <- cumsum(discovered_genes %in% reference_index)
    total_count <- total_count / length(reference_index)
    return(total_count)
}

numReps = 49 #should be 49
cumulative_overlap_counts <- mclapply(0:numReps, function(i) {
    cat(sprintf("Started rep:\t%i, at time %s\n", i, Sys.time()), file="trainReg_progress.txt", append=TRUE)
    
    overlap_counts <- list()
    
    ret <- readDF(i)
    x <- ret$x
    y <- ret$y
    group_index <- ret$group_index
    phenotype_genes <- ret$phenotype_genes
    group_df <- ret$group_df
    
#   ELASTIC NET
    elasticnet <- glmnet(x, y, alpha = 0.600000, lambda = 0.040848)
    coef_enet <- as.matrix(coef(elasticnet, s=elasticnet$lambda))[-1]
    count_enet <- getCounts(coef_enet, phenotype_genes)
    overlap_counts$elastic_net <- count_enet

#   SPARSE GROUP LASSO
    sparse_gl <- SGL(data=list(x=x, y=y), index=group_index, lambda = 0.0175799340984784, type="linear", alpha=0.5)
    coef_sgl  <- sparse_gl$beta
    count_sgl <- getCounts(coef_sgl, phenotype_genes)
    overlap_counts$sparse_gl <- count_sgl

    cat(sprintf("Ended rep:\t%i, at time %s\n", i, Sys.time()), file="trainReg_progress.txt", append=TRUE)
    
    overlap_counts
}, mc.cores=10)

df_from_counts <- function(counts, name) {
    name_list <- lapply(counts, `[[`, name)
    maxlength <- max(lengths(name_list))
    name_list <- lapply(name_list, function(o, m) {
    o <- if (length(o) < m) c(o, integer(m - length(o)) + o[length(o)]) else o
}, maxlength)
    name_list <- do.call('cbind', name_list)
    name_list <- as.data.frame(name_list)
    colnames(name_list) <- paste('run', 1:length(name_list), sep='')
    name_list
}

elastic_net_df <- df_from_counts(cumulative_overlap_counts, 'elastic_net')
sparse_gl_df   <- df_from_counts(cumulative_overlap_counts, 'sparse_gl')

baseSaveDir <- '../../DataFrames/latentRegression/%s'
write.csv(elastic_net_df, file=sprintf(baseSaveDir, 'elastic_net.csv'), row.names=FALSE)
write.csv(sparse_gl_df, file=sprintf(baseSaveDir, 'sparse_gl.csv'), row.names=FALSE)

save(cumulative_overlap_counts, file='/projects/leelab3/psturm/sim_reg_counts.RData')