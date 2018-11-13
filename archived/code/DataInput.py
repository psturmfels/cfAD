import tensorflow as tf

def GetDataSet(batchSize, knownIndices, shuffle=True):
    indicesConst = tf.constant(knownIndices, dtype=tf.int32, name='knownIndices')
    dataset = tf.data.Dataset.from_tensor_slices(indicesConst)
    dataset = dataset.batch(batchSize)
    
    if (shuffle):
        dataset = dataset.shuffle(buffer_size=10000)
        
    return dataset

def CreateIterator(iterType=tf.int32, outputShape=[None, 2]):
    iterator = tf.data.Iterator.from_structure(iterType, outputShape)
    indexBatch = iterator.get_next()
    return iterator, indexBatch

def GetIterInitOp(iter, dataset):
    return iter.make_initializer(dataset)

def GetBatchOperation(expressionMatrix, indexBatch):
    sampleIndexBatchOp  = indexBatch[:, 0]
    traitIndexBatchOp   = indexBatch[:, 1]
    trainDataBatchOp    = tf.gather_nd(expressionMatrix, indexBatch)
    
    return sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp

def CreateSoloOps(expressionMatrix, batchSize, trainIndices):
    with tf.variable_scope('DataPipeline'):
        trainSet = GetDataSet(batchSize, trainIndices)
        iter, indexBatch = CreateIterator()
        trainInitOp = GetIterInitOp(iter, trainSet)

        sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp = GetBatchOperation(expressionMatrix, indexBatch)
    return trainInitOp, sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp
    
def CreateJointOps(expressionMatrix, batchSizeTrain, batchSizeTest, trainIndices, testIndices):
    with tf.variable_scope('DataPipeline'):
        trainSet = GetDataSet(batchSizeTrain, trainIndices)
        testSet  = GetDataSet(batchSizeTest,  testIndices, shuffle=False)

        iter, indexBatch = CreateIterator()
        trainInitOp = GetIterInitOp(iter, trainSet)
        testInitOp  = GetIterInitOp(iter, testSet)

        sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp = GetBatchOperation(expressionMatrix, indexBatch)
    return trainInitOp, testInitOp, sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp