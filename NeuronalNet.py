import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# tf.compat.v1.disable_eager_execution()
import gc; gc.collect()

def cross_entropy(prediction_values, target_values, epsilon=1e-10):
     prediction_values = np.clip(prediction_values, epsilon, 1. - epsilon)
     N = prediction_values.shape[0]
     ce_loss = -np.sum(np.sum(target_values * np.log(prediction_values + 1e-5)))/N
     return ce_loss

def splitData4TrainingTesting ( data, numSamples ):
    
    # n = np.size(data, 0)
    # data = np.delete(data, [ 5, 6, 7], 1)
    
    # Training and test data
    # data_train := 80% of data
    # data_test  := 20% of data
    train_start = 0
    train_end = int(np.floor(0.8*numSamples))
    test_start = train_end + 1
    test_end = numSamples
    
    data_train = data[np.arange(train_start, train_end), :]
    data_test =  data[np.arange(test_start,  test_end ), :]
    
    return data_train, data_test

def trainAndPredict(data, params, period=None):   
    
    acceptableTrainingError = params['acceptableTrainingError']
    numSamples = params['numTrainingSamples']
    data = data.values
    
    data_train, data_test = splitData4TrainingTesting( data, numSamples )

    # Scale the data to [0..1] to make the training faster
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(data_train)
    
    data_trainScaled = scaler.transform(data_train)
    data_testScaled = scaler.transform(data_test)

    
    # Build X and y
    X_train = np.delete(data_trainScaled, 3,axis = 1)  # here we drop close price
    y_train = data_trainScaled[:, 3] # here we use close price
    X_test = np.delete(data_testScaled, 3, axis=1)# here we drop close price
    y_test = data_testScaled[:, 3]   # here we use close price

    # take last row from the real data and after
    # we will give it as input to our trained model to get a prediction
    index4test=-1
    X_lastData = X_test[index4test:index4test+1, ::]

    if index4test == -1 : 
        X_lastData = X_test[-1:, ::]

    X_train= X_train[:-1, :] 
    X_test=X_test[:-1, :]  

    y_train=y_train[:-1]
    y_test=y_test[:-1]

    # n_attributes := "number of Attributes/variables in training data"
    n_attributes = X_train.shape[1]
    
    # Neurons
    # n_neurons_1 = 1024
    # n_neurons_2 = 512
    # n_neurons_3 = 256
    # n_neurons_4 = 128
    
    
    n_neurons_1 = 1024
    n_neurons_2 = 1024
    n_neurons_3 = 1024
    n_neurons_4 = 1024
    
    
    
    # Session
    net = tf.compat.v1.InteractiveSession()
    
    # Placeholder
    X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_attributes])
    Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
    
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
    W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
    bias_out = tf.Variable(bias_initializer([1]))
    
    # Hidden layer
    hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
    hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
    hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
    hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
    
    # Output layer (transpose!)
    out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
    
    # Cost function
    mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))
 
    
    # mse = tf.reduce_mean(tf.compat.v1.metrics.mean_absolute_error(Y, out))
   

    # cross_entropy_loss = cross_entropy(out, Y)

    # Optimizer
    # opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(mse)
    opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(mse)

    # opt = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=0.0001).minimize(total_loss)
 
   
    # Init
    net.run(tf.compat.v1.global_variables_initializer())
    
    # Fit neural net
    batch_size = 256
    mse_train = []
    mse_test = []
    
    stopTraining = False
    # Run
    epochs = 2
    for e in range(epochs):
        
        if (stopTraining == True ) : break
    
        # Shuffle training data
        shuffle_indices = np.random.permutation(np.arange(len(y_train)))
        X_train = X_train[shuffle_indices]
        y_train = y_train[shuffle_indices]
#    
        # Minibatch training
        for i in range(0, len(y_train) // batch_size):
            start = i * batch_size
            batch_x = X_train[start:start + batch_size]
            batch_y = y_train[start:start + batch_size]
            # Run optimizer with batch
            net.run(opt, feed_dict={X: batch_x, Y: batch_y})
    
            # Show progress
            if np.mod(i, 5) == 0:
                mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
                mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
                mse_trainning = mse_test[-1]

                if ( mse_trainning < acceptableTrainingError ):
                    msg = str(mse_trainning) + " at epoch: " + str(e) 
                    print ("DEBUG ::: trainning stopped on error: " + msg )
                    stopTraining = True
                    break

    # Prediction
    predTestData =   net.run(out, feed_dict={ X: X_test })             
    predLastData  =  net.run(out, feed_dict={ X: X_lastData })
    net.close()
   
    n_rowsDataTest, n_colsDataTest = np.shape(data_test)
    predLastData2inverse = np.full( (1, n_colsDataTest),
                                    predLastData[0][0] ) 
    
    predLastDataRescaled = scaler.inverse_transform(predLastData2inverse)

    n_rowsAllPred, n_colsAllPred = np.shape( predTestData )
    predTestData2inverse = np.full( (n_colsAllPred, n_colsDataTest), 0.)
    # we take the 3rd coloumn because it's a close price in the origin!
    predTestData2inverse[:, 3] = np.hstack(predTestData)
    predTestDataRescaled = scaler.inverse_transform( predTestData2inverse)
       
    prediction = {}
    prediction['predTestingData'] = predTestDataRescaled
    prediction['predLastData'] = predLastDataRescaled
    prediction['algorithm'] = __name__ 
    
    return  prediction