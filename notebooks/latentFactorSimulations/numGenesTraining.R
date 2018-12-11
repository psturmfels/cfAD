library('SGL')
library('glmnet')
library('parallel')

numReps <- 49
numSamples <- 500
numPheno <- 1

getCounts <- function(coeffs, reference_index) {
    discovered_genes <- which(coeffs != 0)
    discovered_genes <- discovered_genes[order(abs(coeffs[discovered_genes]), decreasing=TRUE)]
    num_discovered <- length(discovered_genes)
    total_count <- cumsum(discovered_genes %in% reference_index)
    total_count <- total_count / length(reference_index)
    return(total_count)
}

base_dir <- '/projects/leelab3/psturm/simulatedData/varyDimData/g%i/df%i.csv'

cumulative_overlap_counts <- mclapply(0:numReps, function(i) {
    cat(sprintf("Started rep:\t%i, at time %s\n", i, Sys.time()), file="ng_reg_progress.txt", append=TRUE)
    overlap_counts <- list()
    
    for (g in c(1000, 3000, 5000, 7000)) {
        numGenes = g - numPheno
        df = read.csv(sprintf(base_dir, g, i))
        df_t = as.data.frame(t(df))
        colnames(df_t) = c(paste('p', sep='', 1:numPheno), paste('g', sep='', 1:numGenes))
        data_mat = df_t[1:numSamples, ]
        
        numPaths <- length(colnames(df)) - numSamples - 2
        bin_path_mat <- df[paste('pathway', 0:(numPaths-1), sep='')]
        bin_path_mat <- bin_path_mat[-1, ]
        
        group_index <- integer(numGenes) - 1
        group_obj <- which(bin_path_mat == 1, arr.ind=T)
        group_index[as.vector(group_obj[, 1])] <- as.vector(group_obj[, 2])
        
        phenotype_genes <- which(df[['phenotype_genes']][-1] != 0)
        y <- data_mat$p1
        x <- data.matrix(data_mat[, paste('g', sep='', 1:numGenes)])
        
        elasticnet <- glmnet(x, y, alpha = 0.600000, lambda = 0.040848)
        coef_enet <- as.matrix(coef(elasticnet, s=elasticnet$lambda))[-1]
        count_enet <- getCounts(coef_enet, phenotype_genes)
        
        sparse_gl <- SGL(data=list(x=x, y=y), index=group_index, lambda = 0.0175799340984784, type="linear", alpha=0.5)
        coef_sgl  <- sparse_gl$beta
        count_sgl <- getCounts(coef_sgl, phenotype_genes)
        
        overlap_counts[[sprintf('enet_g%i', g)]] <- count_enet
        overlap_counts[[sprintf('sgl_g%i', g)]] <- count_sgl
    }
    cat(sprintf("Ended rep:\t%i, at time %s\n", i, Sys.time()), file="ng_reg_progress.txt", append=TRUE)
    
    overlap_counts
}, mc.cores=10) #TODO FIX ME

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

baseSaveDir <- '../../DataFrames/latentRegression/%s'
for (g in c(1000, 3000, 5000, 7000)) {
    overlap_df <- df_from_counts(cumulative_overlap_counts, sprintf('enet_g%i', g))
    save_name <- sprintf('numGenes_enet_g%i.csv', g)
    write.csv(overlap_df, file=sprintf(baseSaveDir, save_name), row.names=FALSE)
    
    overlap_df <- df_from_counts(cumulative_overlap_counts, sprintf('sgl_g%i', g))
    save_name <- sprintf('numGenes_sgl_g%i.csv', g)
    write.csv(overlap_df, file=sprintf(baseSaveDir, save_name), row.names=FALSE)
}