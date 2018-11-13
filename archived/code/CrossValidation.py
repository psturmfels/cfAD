from sklearn.model_selection import KFold
import tensorflow as tf
import numpy as np
from TrainModel import TrainModel
from BuildModel import *
from DataInput import *

#from joblib import Parallel, delayed
#class DataHolder: pass
#def GetPerfOnKFolds(dataHolder):
#    splitIter = dataHolder.splitIter
#    eta = dataHolder.eta
#    lamb = dataHolder.lamb
#Consider running cross validation in parallel

def CrossValidateParams(X, latentDim, etas, lambs, foldcount=10):
    n, g = X.shape
    expressionMatrix = tf.constant(X, dtype=tf.float32, name='expressionMatrix')

    knownIndices = np.argwhere(~np.isnan(X))
    numKnown, _ = knownIndices.shape
    np.random.shuffle(knownIndices)
    
    kf = KFold(n_splits=foldcount, shuffle=True)
    
    errors = np.zeros((len(etas), len(lambs), foldcount))
    
    batchSizeTrain = 1
    batchSizeTest  = int(numKnown / foldcount) + 1
    
    #Create the data ingestion operations
    with tf.variable_scope('DataPipeline'):
        iter, indexBatch = CreateIterator()
        sampleIndexBatchOp, traitIndexBatchOp, trainDataBatchOp = GetBatchOperation(expressionMatrix, indexBatch)

    #Create some helper operations to set hyper-parameters without having to rebuild model each time
    learningRate = tf.get_variable('learningRate', shape=(), dtype=tf.float32)
    regMult = tf.get_variable('regMultiplier', shape=(), dtype=tf.float32)
    learningRateInput = tf.placeholder(dtype=tf.float32, shape=(), name='learningRateInput')
    regMultInput = tf.placeholder(dtype=tf.float32, shape=(), name='regMultiplierInput')
    assignLearningRateOp = tf.assign(learningRate, learningRateInput)
    assignRegMultOp = tf.assign(regMult, regMultInput)
    
    #Create the model operations
    print('Building the model graph...')
    sampleTraitPredictions, embeddingsRegularizer = GetPredOps(n, g, sampleIndexBatchOp, traitIndexBatchOp, latentDim)
    trainOp, predictionLoss = GetOptOps(sampleTraitPredictions, embeddingsRegularizer, trainDataBatchOp, 
                                        learningRate=learningRate, reg=regMult)
    
    errors = np.zeros((len(etas), len(lambs), foldcount))
    
    with tf.Session() as sess:         
        #Loop through all of the folds
        fold = 0
        for train_index, test_index in kf.split(knownIndices):
            trainIndices = knownIndices[train_index]
            testIndices  = knownIndices[test_index]
    
            #Link iterator to the current indices
            with tf.variable_scope('DataPipeline'):
                trainSet = GetDataSet(batchSizeTrain, trainIndices)
                testSet  = GetDataSet(batchSizeTest,  testIndices, shuffle=False)
                trainInitOp = GetIterInitOp(iter, trainSet)
                testInitOp  = GetIterInitOp(iter, testSet)
            
            lamb_ind = 0
            for lamb in lambs:
                eta_ind = 0
                for eta in etas:
                    sess.run(tf.global_variables_initializer())
                    
                    #Assign the current hyper-parameters
                    learningRate, regMul = sess.run([assignLearningRateOp, assignRegMultOp], 
                             feed_dict={
                                 learningRateInput: eta,
                                 regMultInput: lamb
                             })

                    summaryDir='../summaries/eta{}_lamb{}/fold{}/'.format(eta, lamb, fold)
                    checkpointDir='../checkpoints/eta{}_lamb{}/fold{}/'.format(eta, lamb, fold)
                    
                    
                    #TODO: finish this. Basically, write summaries to some place,
                    #and then collect all of the errors and write that some place to.
                    #Also, consider summarizing the actual latent representations themselves.
                    TrainModel(sess, trainOp, predictionLoss,
                               trainInitOp, testInitOp=testInitOp,
                               numEpochs=1, verbSteps=100, summaryDir=summaryDir, 
                               checkpointDir=checkpointDir)
                    sess.run(testInitOp)
                    testLoss = sess.run(predictionLoss)
                    
                    errors[eta_ind, lamb_ind, fold] = testLoss
                    print('eta={}, lamb={}, fold={}, loss={}'.format(eta, lamb, fold, testLoss))
                    
                    eta_ind += 1
                lamb_ind += 1
            fold += 1
    np.save('cv_errors', errors)
    np.save('etas', np.array(etas))
    np.save('lambs', np.array(lambs))