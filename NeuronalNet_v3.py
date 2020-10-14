# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:43:12 2020

@author: mvereda
"""
import os

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# tf.compat.v1.disable_eager_execution()
import gc; gc.collect()
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

def findPeaksValleys (data_test):
    
    series = data_test[:,3]
    print (series)    
   
    # Find indices of peaks
    peak_idx, _ = find_peaks(series, distance=10)
    
    # Find indices of valleys (from inverting the signal)
    valley_idx, _ = find_peaks(-series, distance=10)
    
    # Plot 
    t = np.arange(start=0, stop=len(series), step=1, dtype=int)
    plt.plot(t, series)   
    
    # Plot peaks (red) and valleys (blue)
    plt.plot(t[peak_idx], series[peak_idx], 'g^')
    plt.plot(t[valley_idx], series[valley_idx], 'rv')
    
    plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
    
    plt.show()    
    
    
def splitData4TrainingTesting ( data ):
    
    # n = np.size(data, 0)
    data = np.delete(data, [ 4,5,6,7,9], 1)
    numRowsData, numColsData = np.shape( data )
    # Training and test data
    # data_train := 80% of data
    # data_test  := 20% of data
    train_start = 0
    train_end = int(np.floor(0.8*numRowsData))
    test_start = train_end + 1
    test_end = numRowsData
    
    data_train = data[np.arange(train_start, train_end), :]
    data_test =  data[np.arange(test_start,  test_end ), :]
    
    return data_train, data_test

def buildX (data, numPastSamples, numfutureSamples):
    
    numAttributes = data.shape[1]
    msg = 'building X matrix of '+ str(numAttributes) + 'x' 
    msg += str(numPastSamples) +' cols per row ...' 
    print (msg)
    
    X_ = np.empty([0, numPastSamples * numAttributes])
    for i in range( numPastSamples, len(data) ):
        newRow = data[ i-numPastSamples : i , :]
        # newRow = data[ i , :]
        newRow = newRow.reshape( (1, numPastSamples * numAttributes ) )       
        X_ = np.append(X_, newRow, axis=0)
        
    X_ =  X_[ : -numfutureSamples , : ]
    
    return X_

# indexAttribute : = 3 cus for TRANSAQ, CLOSE is the 4th coloumn
def buildY( data, numPastSamples, numfutureSamples, indexAttribute=3):
    # Y_ = data[:, indexAttribute]     
    # # it rolls the whole matrix #sizeWindow rows up
    # Y_ = np.roll(Y_, -numfutureSamples, axis=0)
    # # removes last #sizeWindow rows
    # Y_ = Y_[ numPastSamples : -numfutureSamples] 
    
    numPatterns = 1# two patterns := { 2Min-pattern, 5Min-pattern }
    Y_ = np.empty([0,numPatterns])    
    y = data[:, indexAttribute]
    indexes_y = range(numPastSamples, len (y) - numfutureSamples)
    
    for i in indexes_y :
        # Y_ = np.append(Y_, [ [ y[i+2] , y[i+numfutureSamples] ] ], axis=0)
        Y_ = np.append(Y_, [ [ y[i+numfutureSamples] ] ], axis=0)
    
    # Y_ = Y_[ : -numfutureSamples] 

    return Y_

def getLastRow ( data, numPastSamples):
    #TODO hmmmmmm
    numAttributes = data.shape[1] 
    X_lastData = np.empty([0, numPastSamples * numAttributes])
    newRow = data[ -numPastSamples : , :].flatten()
    X_lastData = np.vstack( (X_lastData, newRow) )
    
    return X_lastData

# indexAttribute : = 3 cus for TRANSAQ, CLOSE is the 4th coloumn
def getOriginalScale ( data_test, scaler, dataScaled, indexAttribute=3 ):
    
    n_rowsDataTest,   n_colsDataTest   = np.shape( data_test )
    n_rowsDataScaled, n_colsDataScaled = np.shape( dataScaled )    
    dataOriginalScale = np.empty([n_rowsDataScaled, n_colsDataScaled])
    
    for i in range(n_rowsDataScaled):
        data2inverse = np.full( (n_colsDataScaled, n_colsDataTest), 0.)
        data2inverse[:, indexAttribute] = np.hstack( dataScaled[i] )    
        dataScaledInverted = scaler.inverse_transform( data2inverse )        
        dataOriginalScale[i] = dataScaledInverted[:, indexAttribute]    
    
    return dataOriginalScale

class NeuronalNet_v3_Model:
    
    def __init__(self, df, params, period=None):
        self.model = None
        self.period = period
        self.fileName =  __name__ + "_" + self.period
        self.params = params
        self.out  = None  # output layer
        
        if ( os.path.isdir( self.fileName ) ):
            print ('pre-trainned model found! loading it ...')
            # self.model = load_model( self.fileName ) 
            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()
            
            # Later, launch the model, use the saver to restore variables from disk, and
            # do some work with the model.
            with tf.Session() as net:
              # Restore variables from disk.
              saver.restore(net, self.fileName)
              print("Model restored.")
              # Check the values of the 
              self.model = net
        else:        
            self.load(df)
            
    def load (self, df ):
        
        acceptableTrainingError = self.params['acceptableTrainingError']
        numfutureSamples = int(self.period[0])   # Prediction Window
        numPastSamples = self.params['numPastSamples']
        # indexAttribute : = 3 cus for TRANSAQ, CLOSE is the 4th coloumn
        indexAttribute=3
        data = df.values
     
        data_train, data_test = splitData4TrainingTesting( data )
    
        # Scale the data to [0..1] to make trainning faster
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_train)
        
        data_trainScaled = scaler.transform(data_train)
        data_testScaled = scaler.transform(data_test)
    
        # Build X and y    
        X_train = buildX( data_trainScaled, numPastSamples, numfutureSamples )
        y_train = buildY( data_trainScaled, numPastSamples, numfutureSamples)
            
        X_test  = buildX( data_testScaled, numPastSamples, numfutureSamples )  
        y_test  = buildY( data_testScaled, numPastSamples, numfutureSamples)
    
        # take last row from the real data and after we will give it
        # as input to our already trainned model to get a prediction   
        X_lastData = getLastRow( data_testScaled, numPastSamples)    
    
        # n_attributes := "number of Attributes/coloumns in training data"
        n_attributes = X_train.shape[1]
        n_patterns = y_train.shape[1]
        
        # Neurons
        n_neurons_1 = 1024 #512 1024
        n_neurons_2 = 1024 #256 1024
        n_neurons_3 = 1024 #128 1024
        n_neurons_4 = 1024 #64 1024
    
        
         # Session
        net = tf.compat.v1.InteractiveSession()
        
        # Placeholder
        X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_attributes])
        # Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_patterns ])
        Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[n_patterns, None ])
        
        # Initializers
        sigma = 1
        weight_initializer = tf.compat.v1.variance_scaling_initializer(
            mode="fan_avg", 
            distribution="uniform", 
            scale=sigma
        )
        bias_initializer = tf.zeros_initializer()
        
        # Hidden weights
        W_hidden_1 = tf.Variable(weight_initializer([n_attributes, n_neurons_1]))
        bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
        W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
        bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
        W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
        bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
        W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
        bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
        
        # Output weights
        W_out = tf.Variable(weight_initializer([n_neurons_4, n_patterns]))
        bias_out = tf.Variable(bias_initializer([1,1]))
        
        # Hidden layer
        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
        hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
        hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
        
        # Output layer (transpose!)
        out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
            
         # Cost function
        mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))
        
        # Optimizer
        opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)
        # opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01,
        #                                        beta1=0.9, 
        #                                        beta2=0.999, 
        #                                        epsilon=1e-08).minimize(mse)
     
        # Init
        net.run(tf.compat.v1.global_variables_initializer())
        
        # Fit neural net
        batch_size = 256
        mse_train = []
        mse_test = []    
        stopTraining = False
        epochs = 50
        
        print('Trainning Machine....' )
        
        for e in range(epochs):
            
            if (stopTraining == True ) : break
        
            # Shuffle training data
            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]
    
            # Minibatch training
            for i in range(0, len(y_train) // batch_size):
                start = i * batch_size
                batch_x = X_train[start:start + batch_size]
                batch_y = y_train[start:start + batch_size].transpose()
                
                # Run optimizer with batch
                net.run(opt, feed_dict={X: batch_x, Y: batch_y})
        
                # Show progress
                if np.mod(i, 5) == 0:
                    mse_train.append(
                        net.run( mse, feed_dict={X: X_train, Y: y_train.transpose() } )
                    )
                    mse_test.append(
                        net.run( mse, feed_dict={X: X_test, Y: y_test.transpose()} )
                    )
                    mse_trainning = mse_test[-1]
                    print ("trainning error: " + str(mse_trainning) +' epoch: '+str(e))
                    if ( mse_trainning < acceptableTrainingError ):
                        msg = str(mse_trainning) + " at epoch: " + str(e) 
                        print ("DEBUG ::: trainning stopped on error: " + msg )
                        stopTraining = True
                        break
                    
        saver = tf.train.Saver()   
        save_path = saver.save(net, self.fileName)
        print("Model saved in path: %s" % save_path)
            
        self.model = net
        self.out = out
        
        
    def predict(self, df ):
        
        numPastSamples = self.params['numPastSamples']

        data = df.values
        # indexAttribute : = 3 cus for TRANSAQ, CLOSE is the 4th coloumn
        indexAttribute=3
        # Scale the data to [0..1] to make trainning faster
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)
        
        data_testScaled = scaler.transform(data)
        X_lastData = getLastRow( data_testScaled, numPastSamples)    

        # Prediction
        predLastDataScaled =  self.model.run(self.out, feed_dict={ X: X_lastData })
        self.model.close()   
        
        predLastData = getOriginalScale( data, scaler, predLastDataScaled )    
    
        prediction = {}
        prediction['predLastData'] = predLastData
        prediction['predWindow'] = self.period
        prediction['algorithm'] = __name__ 
        prediction['lastClosePrice'] = data[-1,indexAttribute]
        
        return  prediction
    


def trainAndPredict(data, params, period=None):   
    
    acceptableTrainingError = params['acceptableTrainingError']
    numSamples = params['numTrainingSamples']    
    numfutureSamples = params['numfutureSamples']   # Prediction Window
    numPastSamples = params['numPastSamples']
    # indexAttribute : = 3 cus for TRANSAQ, CLOSE is the 4th coloumn
    indexAttribute=3
    data = data.values
 
    data_train, data_test = splitData4TrainingTesting( data, numSamples )

    # Scale the data to [0..1] to make trainning faster
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)
    
    data_trainScaled = scaler.transform(data_train)
    data_testScaled = scaler.transform(data_test)

    # Build X and y    
    X_train = buildX( data_trainScaled, numPastSamples, numfutureSamples )
    y_train = buildY( data_trainScaled, numPastSamples, numfutureSamples)
        
    X_test  = buildX( data_testScaled, numPastSamples, numfutureSamples )  
    y_test  = buildY( data_testScaled, numPastSamples, numfutureSamples)

    # take last row from the real data and after we will give it
    # as input to our already trainned model to get a prediction   
    X_lastData = getLastRow( data_testScaled, numPastSamples)    

    # n_attributes := "number of Attributes/coloumns in training data"
    n_attributes = X_train.shape[1]
    n_patterns = y_train.shape[1]
    
    # Neurons
    # n_neurons_1 = 512
    # n_neurons_2 = 256
    # n_neurons_3 = 128
    # n_neurons_4 = 64
    n_neurons_1 = 1024
    n_neurons_2 = 1024
    n_neurons_3 = 1024
    n_neurons_4 = 1024

    
     # Session
    net = tf.compat.v1.InteractiveSession()
    
    # Placeholder
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_attributes])
    # Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_patterns ])
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[n_patterns, None ])
    
    # Initializers
    sigma = 1
    weight_initializer = tf.compat.v1.variance_scaling_initializer(
        mode="fan_avg", 
        distribution="uniform", 
        scale=sigma
    )
    bias_initializer = tf.zeros_initializer()
    
    # Hidden weights
    W_hidden_1 = tf.Variable(weight_initializer([n_attributes, n_neurons_1]))
    bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
    W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
    bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
    W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
    bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
    W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
    bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
    
    # Output weights
    #W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
    W_out = tf.Variable(weight_initializer([n_neurons_4, n_patterns]))
    bias_out = tf.Variable(bias_initializer([1,1]))
    
    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    
    # Output layer (transpose!)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
    
     # Cost function
    mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))
    
    # Optimizer
    opt = tf.compat.v1.train.AdamOptimizer().minimize(mse)
    # opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01,
    #                                        beta1=0.9, 
    #                                        beta2=0.999, 
    #                                        epsilon=1e-08).minimize(mse)
 
    # Init
    net.run(tf.compat.v1.global_variables_initializer())
    
    # Fit neural net
    batch_size = 256
    mse_train = []
    mse_test = []    
    stopTraining = False
    epochs = 50
    
    for e in range(epochs):
        
        if (stopTraining == True ) : break
    
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]

        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size].transpose()
            
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    
            # Show progress
            if np.mod(i, 5) == 0:
                mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train.transpose() }))
                mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test.transpose()} ))
                mse_trainning = mse_test[-1]
                print ("trainning error: " + str(mse_trainning) +' epoch: '+str(e))
                if ( mse_trainning < acceptableTrainingError ):
                    msg = str(mse_trainning) + " at epoch: " + str(e) 
                    print ("DEBUG ::: trainning stopped on error: " + msg )
                    stopTraining = True
                    break
    

    # Prediction
    predTestDataScaled =  net.run(out, feed_dict={ X: X_test })             
    predLastDataScaled =  net.run(out, feed_dict={ X: X_lastData })
    net.close()   
    
    predLastData = getOriginalScale( data_test, scaler, predLastDataScaled )    
    predTestData = getOriginalScale( data_test, scaler, predTestDataScaled )

    prediction = {}
    prediction['predTestingData'] = predTestData
    prediction['predLastData'] = predLastData
    prediction['predWindow'] = numfutureSamples
    prediction['algorithm'] = __name__ 
    prediction['lastClosePrice'] = data_test[-1,indexAttribute]
    
    return  prediction