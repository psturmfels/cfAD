library('SGL')
library('grpregOverlap')
library('MASS')
library('glmnet')
library('parallel')

#Read in the data
df <- read.csv('/projects/leelab3/psturm/concatData/totalDataDF.csv', header=TRUE)
binaryPathwayDF <- read.delim('/projects/leelab3/psturm/concatData/pathways.tsv', header=TRUE)

rownames(df) <- df$PCG

df = df [ , !(names(df) %in% c('PCG'))]

binaryPathwayDF$Genes <- unlist(lapply(binaryPathwayDF$Genes, as.character))
rownames(binaryPathwayDF) <- binaryPathwayDF$Genes
binaryPathwayDF = binaryPathwayDF [ , !(names(binaryPathwayDF) %in% c('Genes'))]

df_genes <- df[row.names(binaryPathwayDF), ]
df_genes <- df_genes[rowSums(is.na(df_genes)) != ncol(df_genes), ]
binaryPathwayDF <- binaryPathwayDF[row.names(df_genes), ]

braak_stage <- df['BRAAK', ]
braak_stage <- t(braak_stage)

df_genes <- t(df_genes)

#Impute missing values with mean of gene value over samples
for(i in 1:ncol(df_genes)) {
    df_genes[is.na(df_genes[,i]), i] <- mean(df_genes[,i], na.rm = TRUE)
} 

path_group_df <- apply(binaryPathwayDF, 2, function(x, bp) { 
    row.names(bp)[which(x == 1)]
}, binaryPathwayDF)

#Index data values
x <- df_genes[which(!is.na(braak_stage)),]
y <- braak_stage[which(!is.na(braak_stage))]

#Define optimal hyper-parameters from tuning
alpha_ent  = 0.000000 
lambda_ent = 9.393661
alpha_ogl  = 1.000000
lambda_ogl = 0.012099

#Function to get overlap between two runs
get_coeff_overlap <- function(train_coeffs, vald_coeffs) {
    #This function is wrong. First, make sure we are using the same labels (gene indices/names). Second,
    #we need to be careful about intersection. We cannot use vald as a ground truth, rather we have to count intersection up to a certain index for both vectors
    discovered_train <- which(train_coeffs != 0)
    discovered_train <- discovered_train[order(abs(train_coeffs[discovered_train]), decreasing=TRUE)]
    
    discovered_vald  <- which(vald_coeffs != 0)
    discovered_vald  <- discovered_vald[order(abs(vald_coeffs[discovered_vald]), decreasing=TRUE)]
    
    num_discovered <- length(discovered_train)
    
    total_count <- cumsum(discovered_train %in% discovered_vald)
    total_count <- total_count / length(discovered_vald)
    return(total_count)
}

run_overlap_tests <- function(x, y) {
    num_samples <- dim(x)[1]
    num_genes   <- dim(x)[2]
    half_samples <- as.integer(num_samples / 2)
    
    random_indices <- sample(num_samples)
    train_indices <- random_indices[1:half_samples]
    vald_indices  <- random_indices[(half_samples + 1):num_samples]
    
    train_x <- x[train_indices, ]
    train_y <- y[train_indices]
    
    vald_x  <- x[vald_indices, ]
    vald_y  <- y[vald_indices]
    
    elasticnet_train <- glmnet(train_x, train_y, alpha = alpha_ent, lambda = lambda_ent)
    ent_coef_train   <- as.matrix(coef(elasticnet_train, s=elasticnet_train$lambda))[-1]
    elasticnet_vald  <- glmnet(vald_x, vald_y, alpha = alpha_ent, lambda = lambda_ent)
    ent_coef_vald    <- as.matrix(coef(elasticnet_vald, s=elasticnet_vald$lambda))[-1]
    count_enet <- getCounts(ent_coef_train, ent_coef_vald)
    
    grp_fit_temp <- grpregOverlap(train_x, train_y, path_group_df, penalty="grLasso", alpha=a) #path_group_df needs to be adjusted for the specific scenario
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
    
    overlap_counts$elastic_net <- count_enet

#   SPARSE GROUP LASSO
    sparse_gl <- SGL(data=list(x=x, y=y), index=group_index, lambda = 0.0175799340984784, type="linear", alpha=0.5)
    coef_sgl  <- sparse_gl$beta
    count_sgl <- getCounts(coef_sgl, phenotype_genes)
    overlap_counts$sparse_gl <- count_sgl
    
#   OVERLAPPING GROUP LASSO
#     overlap_gl <- grpregOverlap(x, y, group_df, penalty="grLasso", alpha=1, lambda=0.1)
#     coef_overlap <- as.matrix(overlap_gl$beta)[-1]
#     count_overlap <- getCounts(coef_overlap, phenotype_genes)
#     overlap_counts$overlap_gl <- count_overlap
    
    cat(sprintf("Ended rep:\t%i, at time %s\n", i, Sys.time()), file="trainReg_progress.txt", append=TRUE)
    
    overlap_counts
}, mc.cores=2)