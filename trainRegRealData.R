library('SGL')
library('grpregOverlap')
library('MASS')
library('glmnet')
library('parallel')

#Function to get overlap between two runs
get_coeff_overlap <- function(train_coeffs, vald_coeffs) {
    #This function is wrong. First, make sure we are using the same labels (gene indices/names). Second,
    #we need to be careful about intersection. We cannot use vald as a ground truth, rather we have to count intersection up to a certain index for both vectors
    discovered_train <- which(train_coeffs != 0)
    discovered_train <- discovered_train[order(abs(train_coeffs[discovered_train]), decreasing=TRUE)]
    
    discovered_vald  <- which(vald_coeffs != 0)
    discovered_vald  <- discovered_vald[order(abs(vald_coeffs[discovered_vald]), decreasing=TRUE)]
    
    num_discovered <- length(discovered_train)
    total_count <- numeric(num_discovered)
    total_count[1] = as.numeric(discovered_train[1] == discovered_vald[1])
    for (i in 2:num_discovered) {
        total_count[i] <- total_count[i - 1] + as.numeric(discovered_train[i] %in% discovered_vald[1:min(length(discovered_vald), i)])
        if (i <= length(discovered_vald)) {
            total_count[i] <- total_count[i] + as.numeric(discovered_vald[i] %in% discovered_train[1:(i - 1)])
        }
    }
    
    return(total_count)
}

#Function to get overlap
run_overlap_tests <- function(x, y, path_group_df, alpha_ent, lambda_ent, alpha_ogl, lambda_ogl) {
    num_samples <- dim(x)[1]
    num_genes   <- dim(x)[2]
    half_samples <- as.integer(num_samples / 2)
    numReps <- 100
    
    cumulative_overlap_counts <- lapply(1:numReps, function(i) {
        cat(sprintf("Began rep:\t%i, at time %s\n", i, Sys.time()), file="trainReg_real.txt", append=TRUE)
        overlap_counts <- list()
        
        random_indices <- read.csv(sprintf('/projects/leelab3/psturm/realData/randomIndices/perm%i.csv', i - 1), header=FALSE)$V1 + 1
        random_indices <- random_indices[random_indices <= num_samples]
        
        train_indices <- random_indices[1:half_samples]
        vald_indices  <- random_indices[(half_samples + 1):num_samples]

        train_x <- x[train_indices, ]
        train_y <- y[train_indices]

        vald_x  <- x[vald_indices, ]
        vald_y  <- y[vald_indices]

        start <- Sys.time()
        elasticnet_train <- glmnet(train_x, train_y, alpha = alpha_ent, lambda = lambda_ent)
        ent_coef_train   <- as.matrix(coef(elasticnet_train, s=elasticnet_train$lambda))[-1]
        elasticnet_vald  <- glmnet(vald_x, vald_y, alpha = alpha_ent, lambda = lambda_ent)
        ent_coef_vald    <- as.matrix(coef(elasticnet_vald, s=elasticnet_vald$lambda))[-1]
        count_enet <- get_coeff_overlap(ent_coef_train, ent_coef_vald)
    
        ogl_train <- grpregOverlap(train_x, train_y, path_group_df, penalty="grLasso", alpha=alpha_ogl, lambda=lambda_ogl)
        ogl_coef_train <- as.matrix(ogl_train$beta)[-1]
        ogl_vald <- grpregOverlap(vald_x, vald_y, path_group_df, penalty="grLasso", alpha=alpha_ogl, lambda=lambda_ogl)
        ogl_coef_vald <- as.matrix(ogl_vald$beta)[-1]
        count_ogl <- get_coeff_overlap(ogl_coef_train, ogl_coef_vald)
        end <- Sys.time()
        
        overlap_counts$elastic_net <- count_enet
        overlap_counts$overlap_gl  <- count_ogl
        cat(sprintf("Ended rep:\t%i, at time %s\n", i, Sys.time()), file="trainReg_real.txt", append=TRUE)
        
        overlap_counts
    })
    
    cumulative_overlap_counts
}

df_from_counts <- function(counts, name) {
    name_list <- lapply(counts, `[[`, name)
    maxlength <- max(lengths(name_list))
    name_list <- lapply(name_list, function(o, m) {
    o <- if (length(o) < m) c(o, double(m - length(o)) + o[length(o)]) else o
}, maxlength)
    name_list <- do.call('cbind', name_list)
    name_list <- as.data.frame(name_list)
    colnames(name_list) <- paste('run', 1:length(name_list), sep='')
    name_list
}

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

cumulative_overlap_counts <- run_overlap_tests(x, y, path_group_df, alpha_ent, lambda_ent, alpha_ogl, lambda_ogl)

elastic_net_df <- df_from_counts(cumulative_overlap_counts, 'elastic_net')
overlap_gl_df   <- df_from_counts(cumulative_overlap_counts, 'overlap_gl')

baseSaveDir <- 'DataFrames/RealData/%s'
write.csv(elastic_net_df, file=sprintf(baseSaveDir, 'elastic_net.csv'), row.names=FALSE)
write.csv(overlap_gl_df,  file=sprintf(baseSaveDir, 'overlap_gl.csv'), row.names=FALSE)