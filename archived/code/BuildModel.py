import tensorflow as tf

def GetEmbeddingVectors(userBatch, traitBatch):
    with tf.variable_scope('LatentModel'):
        with tf.variable_scope('LatentFactors', reuse=tf.AUTO_REUSE):
            U = tf.get_variable('U')
            V = tf.get_variable('V')

    sampleEmbeddings = tf.nn.embedding_lookup(U, userBatch,  name = 'sampleEmbedCustom')
    traitEmbeddings  = tf.nn.embedding_lookup(V, traitBatch, name = 'traitEmbedCustom')
    customPred = tf.reduce_sum(tf.multiply(sampleEmbeddings, traitEmbeddings), axis=1,   name='customPred')
    return sampleEmbeddings, traitEmbeddings, customPred
    
def GetPredOps(numSamples, numTraits, userBatch, traitBatch, latentDim, device="/cpu:0"):
    with tf.variable_scope('LatentModel'):
        with tf.device('/cpu:0'):
            
            with tf.variable_scope('LatentFactors', reuse=tf.AUTO_REUSE):
                U = tf.get_variable('U', shape=[numSamples, latentDim], initializer=tf.truncated_normal_initializer(stddev=0.02))
                V = tf.get_variable('V', shape=[numTraits, latentDim], initializer=tf.truncated_normal_initializer(stddev=0.02))

            with tf.variable_scope('VectorEmbeddings'):
                sampleEmbeddings = tf.nn.embedding_lookup(U, userBatch,  name = 'sampleEmbeddings')
                traitEmbeddings  = tf.nn.embedding_lookup(V, traitBatch, name = 'traitEmbeddings')

        with tf.device(device):
            with tf.variable_scope('VectorPredictions'):
                sampleTraitPredictions = tf.reduce_sum(tf.multiply(sampleEmbeddings, traitEmbeddings), axis=1,   name='sampleTraitPredictions')
                embeddingsRegularizer  = tf.add(tf.nn.l2_loss(sampleEmbeddings), tf.nn.l2_loss(traitEmbeddings), name='embeddingsRegularizer')
    
    return sampleTraitPredictions, embeddingsRegularizer

def GetOptOps(sampleTraitPredictions, embeddingsRegularizer, trueSampleTraitValues, learningRate=0.005, reg=0.02, device='/cpu:0'):
    globalStep = tf.train.get_global_step()
    if globalStep is None:
        globalStep = tf.train.create_global_step()
    
    with tf.device(device):
        with tf.variable_scope('ModelOptimization'):
            with tf.variable_scope('MeanSquaredError'):
                predictionLoss = tf.reduce_mean(tf.square(tf.subtract(sampleTraitPredictions, trueSampleTraitValues)))
            
            lossFunctionOp = tf.add(predictionLoss, tf.multiply(reg, embeddingsRegularizer), name='lossFunction')
            
            trainOp        = tf.train.GradientDescentOptimizer(learningRate).minimize(lossFunctionOp, global_step=globalStep)
            
    return trainOp, predictionLoss
