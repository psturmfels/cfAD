from CrossValidation import *
import tensorflow as tf
import numpy as np
from BuildModel import *
from DataInput import *
from TrainModel import *

def InputTests():
    n = 100
    g = 1000
    X = np.random.randint(0, 100, (n, g)).astype(np.float32) #some random data, for testing purposes
    X[X < 10] = np.nan
    expressionMatrix = tf.constant(X, dtype=tf.float32, name='expressionMatrix')
    
    batchSizeTrain = 5 #some default constants, for testing purposes
    batchSizeTest  = 10
    iters = 1000
    
    knownIndices = np.argwhere(~np.isnan(X))
    numberKnown, _ = knownIndices.shape
    trainIndices = knownIndices[:int(numberKnown/2), :]
    testIndices  = knownIndices[int(numberKnown/2):, :]
    
    trainInitOp, testInitOp, sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp =\
        CreateJointOps(expressionMatrix, batchSizeTrain, batchSizeTest, trainIndices, testIndices)
    
    with tf.Session() as sess:
        
        for i in range(iters):
            print('Batch {} out of {}'.format(i + 1, iters), end='\r')
            
            sess.run(trainInitOp)
            sampleIndices, traitIndices, dataValues = sess.run([sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp])
            assert np.all(X[sampleIndices, traitIndices] == dataValues), "Assertion failed. dataValues = {}, but X values = {}".format(dataValues, X[sampleIndices, traitIndices])
            
            sess.run(testInitOp)
            sampleIndices, traitIndices, dataValues = sess.run([sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp])
            assert np.all(X[sampleIndices, traitIndices] == dataValues), "Assertion failed. dataValues = {}, but X values = {}".format(dataValues, X[sampleIndices, traitIndices])
            
    print("Input tests passed.")


def TrainingTests():
    n = 10
    g = 100
    latentDim = 5
    
    #Create some random, low-rank data
    U = np.random.randn(n, latentDim).astype(np.float32)
    V = np.random.randn(g, latentDim).astype(np.float32)
    X = np.dot(U, V.T)
    
    expressionMatrix = tf.constant(X, dtype=tf.float32, name='expressionMatrix')
    knownIndices = np.argwhere(~np.isnan(X))
    #For testing purposes, we need to shuffle the indices. If we do not,
    #we will be training on a chunk of the upper half of the matrix, but
    #testing on the lower half of the matrix. This makes no sense,
    #because we don't have any information about the latent variables we are testing on. Therefore, shuffle!
    np.random.shuffle(knownIndices)
    
    numberTestIndices = 200
    
    testIndices  = knownIndices[:numberTestIndices, :]
    trainIndices = knownIndices[numberTestIndices:, :]
    batchSizeTrain = 1
    batchSizeTest  = numberTestIndices
    
    #Create the data ingestion operations
    trainInitOp, testInitOp, sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp =\
        CreateJointOps(expressionMatrix, batchSizeTrain, batchSizeTest, trainIndices, testIndices)
    
    #Create the model operations
    sampleTraitPredictions, embeddingsRegularizer = GetPredOps(n, g, sampleIndexBatchOp, traitIndexBatchOp, latentDim)
    trainOp, predictionLoss = GetOptOps(sampleTraitPredictions, embeddingsRegularizer, trainDataBatchOp)
    
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        TrainModel(sess, trainOp, predictionLoss,
               trainInitOp, testInitOp=testInitOp,
               numEpochs=20,
               device='/cpu:0', verbSteps=100,
               summaryDir='../summaries/', checkpointDir='../checkpoints/model.ckpt')
        sess.run(testInitOp)
        testLoss = sess.run(predictionLoss)
    print("Trained successfully with a final test loss of {}".format(testLoss))
    
def CrossValidationTests():
    n = 10
    g = 100
    latentDim = 5

    #Create some random, low-rank data
    U = np.random.randn(n, latentDim).astype(np.float32)
    V = np.random.randn(g, latentDim).astype(np.float32)
    X = np.dot(U, V.T)
    etas = [0.01, 0.005, 0.001]
    lambs = [0.05, 0.02, 0.01]
    CrossValidateParams(X, latentDim, etas, lambs, foldcount=10)
    
CrossValidationTests()