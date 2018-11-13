import tensorflow as tf
import numpy as np
    
def TrainModel(sess, trainOp, predictionLoss,
               trainInitOp, testInitOp=None,
               numEpochs=200, numEpochsEarlyStop=20,
               device='/cpu:0', verbSteps=None,
               summaryDir=None, checkpointDir=None,
               restore=False):
    
    #Set up writers to plot the loss over time on the training and test sets
    if summaryDir is not None:
        lossSummary = tf.summary.scalar("Prediction Loss", predictionLoss)
        trainWriter = tf.summary.FileWriter(summaryDir + 'train/', sess.graph)
        testWriter  = tf.summary.FileWriter(summaryDir + 'test/')
        summaryOp   = tf.summary.merge_all()
        
    #Restore the model if desired
    if checkpointDir is not None:
        saver = tf.train.Saver()
        import os
        if not os.path.exists(checkpointDir):
            os.makedirs(checkpointDir)
        if restore:
            saver.restore(sess, checkpointDir)
    
    testLoss = '?'
    bestTestLoss = np.inf
    epochsSinceBest = 0
    
    for epoch in range(numEpochs):
        sess.run(trainInitOp)
        j = 0
        
        #Iterate through an epoch of training
        while True:
            j = j + 1
            try:
                if summaryDir is not None:
                    summaryTrain, trainLoss, _ = sess.run([summaryOp, predictionLoss, trainOp])
                else:
                    trainLoss, _ = sess.run([predictionLoss, trainOp])
                
                if verbSteps is not None and j % verbSteps == 0:
                    print('Epoch {}/{}, batch {}, training loss = {}, test loss = {}'.format(epoch, numEpochs, j, trainLoss, testLoss), end='\r')
            except tf.errors.OutOfRangeError:
                break
        
        #Summarize the training error
        if summaryDir is not None:
            trainWriter.add_summary(summaryTrain, epoch)
            
        #Summarize the test error, if desired
        if testInitOp is not None:
            sess.run(testInitOp)
            
            if summaryDir is not None:
                summaryTest, testLoss = sess.run([summaryOp, predictionLoss])
                testWriter.add_summary(summaryTest, epoch)
            else:
                testLoss = sess.run(predictionLoss)
                
                #Stop training if test loss hasn't improved in numEpochsEarlyStop epochs
                if bestTestLoss > testLoss:
                    bestTestLoss = testLoss
                    epochsSinceBest = 0
                else:
                    epochsSinceBest += 1
                    if epochsSinceBest > numEpochsEarlyStop:
                        print('Reached early stopping criteria. Performance has not improved for {} epochs.'.format(numEpochsEarlyStop))
                        break
                
            print('Epoch {}/{}, batch {}, training loss = {}, test loss = {}'.format(epoch, numEpochs, j, trainLoss, testLoss), end='\r')
        
    #Save the model to a the checkpoint, if desired
    if checkpointDir is not None:
        saver.save(sess, checkpointDir)    
    
    return
