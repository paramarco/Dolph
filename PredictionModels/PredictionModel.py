import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import gc
from scipy.signal import find_peaks
import scipy.signal as signal
import pyampd.ampd as ampd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, LSTM
import os
import copy
import joblib
import pandas as pd
from abc import ABC, abstractmethod
import logging
from PredictionModels.StochasticAndRSIModel import StochasticAndRSIModel

gc.collect()
log = logging.getLogger("PredictionModel")

# Set matplotlib logging level to WARNING (suppress DEBUG and INFO)
matplotlib_logger = logging.getLogger('matplotlib')
matplotlib_logger.setLevel(logging.WARNING)


class Predictions:
    def __init__(self, predictions, training_set):
        self.predictions = predictions
        self.training_set = training_set

def initPredictionModel(data, security ):
    
    params = security["params"]
    
    alg = params['algorithm']
    if alg == 'NeuronalNet':
        return NeuronalNetModel()
    elif alg == 'NeuronalNet_v2':
        return NeuronalNetV2Model()
    elif alg == 'NeuronalNet_v3':
        return NeuronalNetV3Model()
    elif alg == 'NeuronalNet_v5':
        return NeuronalNetV5Model()
    elif alg == 'NeuronalNet_v6':
        return NeuronalNetV6Model()
    elif alg == 'NeuronalNet_v9':
        return NeuronalNetV9Model()
    elif alg == 'NeuronalNet_v10':
        return NeuronalNetV10Model(data, security)
    elif alg == 'peaks_and_valleys':
        return PeaksAndValleysModel(data, security)
    elif alg == 'stochastic_and_rsi':
        return StochasticAndRSIModel(data, security)
    else:
        raise ValueError(f"Algorithm '{alg}' not recognized")


class PredictionModel(ABC):
    
    def __init__(self, df, params):
        self.params = params
        self.period = params['period']
        self.df = df
        self.numPastSamples = params['numPastSamples']
        self.numfutureSamples = params['numfutureSamples']
        self.indexAttribute = params.get('indexAttribute', 3)
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 256)
        self.model_path = params.get('model_path', 'model_v10.h5')
        self.scaler_path = params.get('scaler_path', 'scaler_v10.pkl')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.build_model()

    @abstractmethod
    def build_model(self):
        pass
        # self.model = Sequential()
        # self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.numPastSamples, 1)))
        # self.model.add(LSTM(units=50, return_sequences=False))
        # self.model.add(Dense(units=25))
        # self.model.add(Dense(units=1))
        # self.model.compile(optimizer='adam', loss='mean_squared_error')

    @abstractmethod
    def train(self):
        pass
        # X_train, y_train = self.prepare_data()
        # self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        # self.model.save(self.model_path)
        # joblib.dump(self.scaler, self.scaler_path)

    @abstractmethod
    def load_trained_model(self, security):
        pass
        # if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
        #     self.model = tf.keras.models.load_model(self.model_path)
        #     self.scaler = joblib.load(self.scaler_path)
        #     return True
        # return False
    
    @abstractmethod
    def predict(self, data, sec, period):
        pass
        # if not self.load_trained_model():
        #     self.train()
        # prediction = self.model.predict(X_future)
        # return prediction
    

# NeuronalNet.py
class NeuronalNetModel:
    def __init__(self):
        pass

    def cross_entropy(self, prediction_values, target_values, epsilon=1e-10):
        prediction_values = np.clip(prediction_values, epsilon, 1. - epsilon)
        N = prediction_values.shape[0]
        ce_loss = -np.sum(np.sum(target_values * np.log(prediction_values + 1e-5)))/N
        return ce_loss

    def splitData4TrainingTesting(self, data, numSamples):
        train_start = 0
        train_end = int(np.floor(0.8*numSamples))
        test_start = train_end + 1
        test_end = numSamples
        data_train = data[np.arange(train_start, train_end), :]
        data_test = data[np.arange(test_start, test_end), :]
        return data_train, data_test

    def trainAndPredict(self, data, params, period=None):   
        acceptableTrainingError = params['acceptableTrainingError']
        numSamples = params['numTrainingSamples']
        data = data.values
        
        data_train, data_test = self.splitData4TrainingTesting(data, numSamples)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_train)
        
        data_trainScaled = scaler.transform(data_train)
        data_testScaled = scaler.transform(data_test)

        X_train = np.delete(data_trainScaled, 3, axis=1)
        y_train = data_trainScaled[:, 3]
        X_test = np.delete(data_testScaled, 3, axis=1)
        y_test = data_testScaled[:, 3]

        index4test = -1
        X_lastData = X_test[index4test:index4test+1, ::]

        if index4test == -1:
            X_lastData = X_test[-1:, ::]

        X_train = X_train[:-1, :]
        X_test = X_test[:-1, :]
        y_train = y_train[:-1]
        y_test = y_test[:-1]

        n_attributes = X_train.shape[1]
        
        n_neurons_1 = 1024
        n_neurons_2 = 1024
        n_neurons_3 = 1024
        n_neurons_4 = 1024
        
        net = tf.compat.v1.InteractiveSession()
        X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_attributes])
        Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
        
        sigma = 1
        weight_initializer = tf.compat.v1.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()
        
        W_hidden_1 = tf.Variable(weight_initializer([n_attributes, n_neurons_1]))
        bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
        W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
        bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
        W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
        bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
        W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
        bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
        
        W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
        bias_out = tf.Variable(bias_initializer([1]))
        
        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
        hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
        hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
        
        out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
        mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(mse)

        net.run(tf.compat.v1.global_variables_initializer())
        
        batch_size = 256
        mse_train = []
        mse_test = []
        
        stopTraining = False
        epochs = 2
        for e in range(epochs):
            if stopTraining: break
            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]
            for i in range(0, len(y_train) // batch_size):
                start = i * batch_size
                batch_x = X_train[start:start + batch_size]
                batch_y = y_train[start:start + batch_size]
                net.run(opt, feed_dict={X: batch_x, Y: batch_y})
                if np.mod(i, 5) == 0:
                    mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
                    mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
                    mse_trainning = mse_test[-1]
                    if mse_trainning < acceptableTrainingError:
                        print(f"DEBUG ::: trainning stopped on error: {mse_trainning} at epoch: {e}")
                        stopTraining = True
                        break

        predTestData = net.run(out, feed_dict={X: X_test})
        predLastData = net.run(out, feed_dict={X: X_lastData})
        net.close()
       
        n_rowsDataTest, n_colsDataTest = np.shape(data_test)
        predLastData2inverse = np.full((1, n_colsDataTest), predLastData[0][0]) 
        predLastDataRescaled = scaler.inverse_transform(predLastData2inverse)

        n_rowsAllPred, n_colsAllPred = np.shape(predTestData)
        predTestData2inverse = np.full((n_colsAllPred, n_colsDataTest), 0.)
        predTestData2inverse[:, 3] = np.hstack(predTestData)
        predTestDataRescaled = scaler.inverse_transform(predTestData2inverse)
           
        prediction = {
            'predTestingData': predTestDataRescaled,
            'predLastData': predLastDataRescaled,
            'algorithm': __name__
        }
        
        return prediction


# NeuronalNet_v2.py
class NeuronalNetV2Model:
    def findPeaksValleys(self, data_test):
        series = data_test[:,3]
        peak_idx, _ = find_peaks(series, distance=10)
        valley_idx, _ = find_peaks(-series, distance=10)

        t = np.arange(start=0, stop=len(series), step=1, dtype=int)
        plt.plot(t, series)   
        plt.plot(t[peak_idx], series[peak_idx], 'g^')
        plt.plot(t[valley_idx], series[valley_idx], 'rv')
        plt.gca().xaxis.set_minor_locator(plt.MultipleLocator(1))
        plt.show()

    def splitData4TrainingTesting(self, data, numSamples):
        train_start = 0
        train_end = int(np.floor(0.8*numSamples))
        test_start = train_end + 1
        test_end = numSamples
        data_train = data[np.arange(train_start, train_end), :]
        data_test = data[np.arange(test_start, test_end), :]
        return data_train, data_test

    def buildX(self, data, numPastSamples, numfutureSamples):
        numAttributes = data.shape[1]
        X_ = np.empty([0, numPastSamples * numAttributes])
        for i in range(numPastSamples, len(data)):
            newRow = data[i-numPastSamples : i , :].reshape((1, numPastSamples * numAttributes))       
            X_ = np.append(X_, newRow, axis=0)
        X_ =  X_[: -numfutureSamples , : ]
        return X_

    def buildY(self, data, numPastSamples, numfutureSamples, indexAttribute=3):
        Y_ = data[:, indexAttribute]     
        Y_ = np.roll(Y_, -numfutureSamples, axis=0)
        if numPastSamples == 0:
            Y_ = Y_[: -numfutureSamples]
        else:
            Y_ = Y_[numPastSamples : -numfutureSamples]  
        return Y_

    def getLastRow(self, data, numPastSamples):
        numAttributes = data.shape[1]
        X_lastData = None
        if numPastSamples == 0:
            X_lastData = data[-1:, ::]
        else:
            X_lastData = np.empty([0, numPastSamples * numAttributes])
            newRow = data[-numPastSamples:, :].flatten()
            X_lastData = np.vstack((X_lastData, newRow))
        return X_lastData

    def getOriginalScale(self, data_test, scaler, dataScaled, indexAttribute=3):
        n_rowsDataTest, n_colsDataTest = np.shape(data_test)
        n_rowsDataScaled, n_colsDataScaled = np.shape(dataScaled)
        data2inverse = np.full((n_colsDataScaled, n_colsDataTest), 0.)
        data2inverse[:, indexAttribute] = np.hstack(dataScaled)
        dataScaledInverted = scaler.inverse_transform(data2inverse)
        dataOriginalScale = np.full((n_rowsDataScaled, n_colsDataScaled), dataScaledInverted[:, indexAttribute])
        return dataOriginalScale

    def trainAndPredict(self, data, params, period=None):
        acceptableTrainingError = params['acceptableTrainingError']
        numSamples = params['numTrainingSamples']
        numfutureSamples = params['numfutureSamples']
        numPastSamples = params['numPastSamples']
        indexAttribute = 3
        data = data.values
        
        data_train, data_test = self.splitData4TrainingTesting(data, numSamples)

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data_train)
        
        data_trainScaled = scaler.transform(data_train)
        data_testScaled = scaler.transform(data_test)

        X_train = self.buildX(data_trainScaled, numPastSamples, numfutureSamples)
        y_train = self.buildY(data_trainScaled, numPastSamples, numfutureSamples)
        X_test = self.buildX(data_testScaled, numPastSamples, numfutureSamples)
        y_test = self.buildY(data_testScaled, numPastSamples, numfutureSamples)

        X_lastData = self.getLastRow(data_testScaled, numPastSamples)

        n_attributes = X_train.shape[1]
        
        n_neurons_1 = 1024
        n_neurons_2 = 1024
        n_neurons_3 = 1024
        n_neurons_4 = 1024
        
        net = tf.compat.v1.InteractiveSession()
        X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, n_attributes])
        Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])
        
        sigma = 1
        weight_initializer = tf.compat.v1.variance_scaling_initializer(mode="fan_avg", distribution="uniform", scale=sigma)
        bias_initializer = tf.zeros_initializer()
        
        W_hidden_1 = tf.Variable(weight_initializer([n_attributes, n_neurons_1]))
        bias_hidden_1 = tf.Variable(bias_initializer([n_neurons_1]))
        W_hidden_2 = tf.Variable(weight_initializer([n_neurons_1, n_neurons_2]))
        bias_hidden_2 = tf.Variable(bias_initializer([n_neurons_2]))
        W_hidden_3 = tf.Variable(weight_initializer([n_neurons_2, n_neurons_3]))
        bias_hidden_3 = tf.Variable(bias_initializer([n_neurons_3]))
        W_hidden_4 = tf.Variable(weight_initializer([n_neurons_3, n_neurons_4]))
        bias_hidden_4 = tf.Variable(bias_initializer([n_neurons_4]))
        
        W_out = tf.Variable(weight_initializer([n_neurons_4, 1]))
        bias_out = tf.Variable(bias_initializer([1]))
        
        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
        hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
        hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))
        
        out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
        mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-08).minimize(mse)

        net.run(tf.compat.v1.global_variables_initializer())
        
        batch_size = 256
        mse_train = []
        mse_test = []
        
        stopTraining = False
        epochs = 50
        for e in range(epochs):
            if stopTraining: break
            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]
            for i in range(0, len(y_train) // batch_size):
                start = i * batch_size
                batch_x = X_train[start:start + batch_size]
                batch_y = y_train[start:start + batch_size]
                net.run(opt, feed_dict={X: batch_x, Y: batch_y})
                if np.mod(i, 5) == 0:
                    mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
                    mse_test.append(net.run(mse, feed_dict={X: X_test, Y: y_test}))
                    mse_trainning = mse_test[-1]
                    print("trainning error: " + str(mse_trainning) +' epoch: '+str(e))
                    if mse_trainning < acceptableTrainingError:
                        msg = str(mse_trainning) + " at epoch: " + str(e)
                        print("DEBUG ::: trainning stopped on error: " + msg)
                        stopTraining = True
                        break

        predTestDataScaled = net.run(out, feed_dict={X: X_test})
        predLastDataScaled = net.run(out, feed_dict={X: X_lastData})
        net.close()

        predLastData = self.getOriginalScale(data_test, scaler, predLastDataScaled)
        predTestData = self.getOriginalScale(data_test, scaler, predTestDataScaled)

        prediction = {
            'predTestingData': predTestData,
            'predLastData': predLastData,
            'predWindow': numfutureSamples,
            'algorithm': __name__,
            'lastClosePrice': data_test[-1, indexAttribute]
        }

        return prediction

# NeuronalNet_v3.py
class NeuronalNetV3Model:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.numSamples = params['numSamples']
        self.numfutureSamples = params['numfutureSamples']
        self.numPastSamples = params['numPastSamples']
        self.indexAttribute = params['indexAttribute']
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 256)
        self.learning_rate = params.get('learning_rate', 0.001)

    def build_X(self, data):
        numAttributes = data.shape[1]
        X_ = np.empty([0, self.numPastSamples * numAttributes])
        for i in range(self.numPastSamples, len(data)):
            newRow = data[i-self.numPastSamples : i, :].reshape((1, self.numPastSamples * numAttributes))
            X_ = np.append(X_, newRow, axis=0)
        X_ = X_[: -self.numfutureSamples, :]
        return X_

    def build_Y(self, data):
        Y_ = data[:, self.indexAttribute]
        Y_ = np.roll(Y_, -self.numfutureSamples, axis=0)
        Y_ = Y_[self.numPastSamples : -self.numfutureSamples]
        return Y_

    def train_and_predict(self):
        data = self.df.values
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler.fit(data)

        data_scaled = scaler.transform(data)
        X_train = self.build_X(data_scaled)
        y_train = self.build_Y(data_scaled)

        numAttributes = X_train.shape[1]
        net = tf.compat.v1.InteractiveSession()
        X = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None, numAttributes])
        Y = tf.compat.v1.placeholder(dtype=tf.float32, shape=[None])

        n_neurons_1 = 1024
        n_neurons_2 = 1024
        n_neurons_3 = 1024
        n_neurons_4 = 1024

        W_hidden_1 = tf.Variable(tf.compat.v1.variance_scaling_initializer()([numAttributes, n_neurons_1]))
        bias_hidden_1 = tf.Variable(tf.zeros([n_neurons_1]))
        W_hidden_2 = tf.Variable(tf.compat.v1.variance_scaling_initializer()([n_neurons_1, n_neurons_2]))
        bias_hidden_2 = tf.Variable(tf.zeros([n_neurons_2]))
        W_hidden_3 = tf.Variable(tf.compat.v1.variance_scaling_initializer()([n_neurons_2, n_neurons_3]))
        bias_hidden_3 = tf.Variable(tf.zeros([n_neurons_3]))
        W_hidden_4 = tf.Variable(tf.compat.v1.variance_scaling_initializer()([n_neurons_3, n_neurons_4]))
        bias_hidden_4 = tf.Variable(tf.zeros([n_neurons_4]))

        W_out = tf.Variable(tf.compat.v1.variance_scaling_initializer()([n_neurons_4, 1]))
        bias_out = tf.Variable(tf.zeros([1]))

        hidden_1 = tf.nn.relu(tf.add(tf.matmul(X, W_hidden_1), bias_hidden_1))
        hidden_2 = tf.nn.relu(tf.add(tf.matmul(hidden_1, W_hidden_2), bias_hidden_2))
        hidden_3 = tf.nn.relu(tf.add(tf.matmul(hidden_2, W_hidden_3), bias_hidden_3))
        hidden_4 = tf.nn.relu(tf.add(tf.matmul(hidden_3, W_hidden_4), bias_hidden_4))

        out = tf.transpose(tf.add(tf.matmul(hidden_4, W_out), bias_out))
        mse = tf.reduce_mean(tf.compat.v1.squared_difference(out, Y))
        opt = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(mse)

        net.run(tf.compat.v1.global_variables_initializer())

        mse_train = []
        stopTraining = False

        for e in range(self.epochs):
            if stopTraining: break
            shuffle_indices = np.random.permutation(np.arange(len(y_train)))
            X_train = X_train[shuffle_indices]
            y_train = y_train[shuffle_indices]
            for i in range(0, len(y_train) // self.batch_size):
                start = i * self.batch_size
                batch_x = X_train[start:start + self.batch_size]
                batch_y = y_train[start:start + self.batch_size]
                net.run(opt, feed_dict={X: batch_x, Y: batch_y})
                if np.mod(i, 5) == 0:
                    mse_train.append(net.run(mse, feed_dict={X: X_train, Y: y_train}))
                    mse_trainning = mse_train[-1]
                    if mse_trainning < self.params['acceptableTrainingError']:
                        print(f"DEBUG ::: training stopped on error: {mse_trainning} at epoch: {e}")
                        stopTraining = True
                        break

        X_lastData = self.build_X(data_scaled[-self.numPastSamples:])
        predLastData = net.run(out, feed_dict={X: X_lastData})
        net.close()

        predLastDataRescaled = scaler.inverse_transform(np.hstack((np.zeros_like(data_scaled), predLastData)))
        prediction = {
            'predLastData': predLastDataRescaled[:, self.indexAttribute],
            'algorithm': __name__
        }

        return prediction

# NeuronalNet_v4.py
class NeuronalNetV4Model:
    def __init__(self, df, params):
        self.dates = df.index.tolist()
        self.loadData(df)
        self.numAttributes = self.df_scaled.shape[1]
        self.epoch = params.get('epochs', 500)
        self.num_layers = params.get('num_layers', 1)
        self.size_layer = params.get('size_layer', 128)
        self.forget_bias = params.get('forget_bias', 0.8)
        self.learning_rate = params.get('learning_rate', 0.005)

        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple=False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(self.size_layer) for _ in range(self.num_layers)],
            state_is_tuple=False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, self.numAttributes))
        self.Y = tf.placeholder(tf.float32, (None, self.numAttributes))
        drop = tf.contrib.rnn.DropoutWrapper(rnn_cells, output_keep_prob=self.forget_bias)
        self.hidden_layer = tf.placeholder(tf.float32, (None, self.num_layers * 2 * self.size_layer))
        self.outputs, self.last_state = tf.nn.dynamic_rnn(drop, self.X, initial_state=self.hidden_layer, dtype=tf.float32)
        self.logits = tf.layers.dense(self.outputs[-1], self.numAttributes, kernel_initializer=tf.glorot_uniform_initializer())
        self.cost = tf.reduce_mean(tf.square(self.Y - self.logits))
        self.optimizer = tf.train.AdamOptimizer(self.learning_rate).minimize(self.cost)
        self.scaler = None
        self.df = None
        self.df_scaled = None
        self.sess = None
        
        tf.reset_default_graph()
        self.sess = tf.InteractiveSession()
        self.sess.run(tf.global_variables_initializer())

    def loadData(self, df):
        listAttributes = ['Polarity', 'Sensitivity', 'Tweet_vol', 'Close_Price']
        self.df = df
        self.scaler = MinMaxScaler().fit(df[listAttributes].astype('float32'))
        df_scaled = self.scaler.transform(df[listAttributes].astype('float32'))
        self.df_scaled = pd.DataFrame(df_scaled)

    def train(self):
        numSamples = self.df_scaled.shape[0]
        ts = 5
        
        for i in range(self.epoch):
            init_value = np.zeros((1, self.num_layers * 2 * self.size_layer))
            total_loss = 0
            for k in range(0, (numSamples // ts) * ts, ts):
                batch_x = np.expand_dims(self.df_scaled.iloc[k : k + ts].values, axis=0)
                batch_y = self.df_scaled.iloc[k + 1 : k + ts + 1].values
                last_state, x, loss = self.sess.run(
                    [self.last_state, self.optimizer, self.cost],
                    feed_dict={
                        self.X: batch_x,
                        self.Y: batch_y,
                        self.hidden_layer: init_value,
                    },
                )
                init_value = last_state
                total_loss += loss
            total_loss /= numSamples // ts
            if (i + 1) % 100 == 0:
                print(f"epoch: {i + 1}, avg loss: {total_loss}")

    def predict(self, df_future):
        future_dates = df_future.index.tolist()
        result = []
        output = np.zeros((1, self.num_layers * 2 * self.size_layer))
        for i in range(len(df_future)):
            out_logits, output = self.sess.run(
                [self.logits, self.last_state],
                feed_dict={
                    self.X: np.expand_dims(self.df_scaled.iloc[i:i + 1].values, axis=0),
                    self.hidden_layer: output,
                },
            )
            result.append(out_logits[0, 0])
            df_future.iloc[i, :] = out_logits[0]
        return pd.DataFrame(result, index=future_dates, columns=['Close_Price'])

    def getPrediction(self, df_future):
        self.train()
        return self.predict(df_future)


# NeuronalNetV5Model
class NeuronalNetV5Model:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.numPastSamples = params['numPastSamples']
        self.numfutureSamples = params['numfutureSamples']
        self.indexAttribute = params.get('indexAttribute', 3)
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 256)
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.numPastSamples, 1)))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def prepare_data(self):
        data = self.df.values
        data_scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.numPastSamples, len(data_scaled) - self.numfutureSamples):
            X.append(data_scaled[i - self.numPastSamples:i, self.indexAttribute])
            y.append(data_scaled[i + self.numfutureSamples, self.indexAttribute])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def train(self):
        X_train, y_train = self.prepare_data()
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs)

    def predict(self, df_future):
        data_future = df_future.values
        data_future_scaled = self.scaler.transform(data_future)
        X_future = []
        for i in range(self.numPastSamples, len(data_future_scaled)):
            X_future.append(data_future_scaled[i - self.numPastSamples:i, self.indexAttribute])
        X_future = np.array(X_future)
        X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
        pred_future_scaled = self.model.predict(X_future)
        pred_future = self.scaler.inverse_transform(np.hstack((np.zeros_like(data_future_scaled), pred_future_scaled)))
        return pred_future[:, self.indexAttribute]

    def getPrediction(self, df_future):
        self.train()
        return self.predict(df_future)




class NeuronalNetV6Model:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.numPastSamples = params['numPastSamples']
        self.numfutureSamples = params['numfutureSamples']
        self.indexAttribute = params.get('indexAttribute', 3)
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 256)
        self.model_path = params.get('model_path', 'model.h5')
        self.scaler_path = params.get('scaler_path', 'scaler.pkl')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=50, return_sequences=True, input_shape=(self.numPastSamples, 1)))
        self.model.add(LSTM(units=50, return_sequences=False))
        self.model.add(Dense(units=25))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def prepare_data(self):
        data = self.df.values
        data_scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.numPastSamples, len(data_scaled) - self.numfutureSamples):
            X.append(data_scaled[i - self.numPastSamples:i, self.indexAttribute])
            y.append(data_scaled[i + self.numfutureSamples, self.indexAttribute])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def train(self):
        X_train, y_train = self.prepare_data()
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_trained_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict(self, df_future):
        if not self.load_trained_model():
            self.train()
        
        data_future = df_future.values
        data_future_scaled = self.scaler.transform(data_future)
        X_future = []
        for i in range(self.numPastSamples, len(data_future_scaled)):
            X_future.append(data_future_scaled[i - self.numPastSamples:i, self.indexAttribute])
        X_future = np.array(X_future)
        X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
        pred_future_scaled = self.model.predict(X_future)
        pred_future = self.scaler.inverse_transform(np.hstack((np.zeros_like(data_future_scaled), pred_future_scaled)))
        return pred_future[:, self.indexAttribute]

    def getPrediction(self, df_future):
        return self.predict(df_future)


# NeuronalNetV9Model
class NeuronalNetV9Model:
    def __init__(self, df, params):
        self.df = df
        self.params = params
        self.numPastSamples = params['numPastSamples']
        self.numfutureSamples = params['numfutureSamples']
        self.indexAttribute = params.get('indexAttribute', 3)
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 256)
        self.model_path = params.get('model_path', 'model_v9.h5')
        self.scaler_path = params.get('scaler_path', 'scaler_v9.pkl')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=60, return_sequences=True, input_shape=(self.numPastSamples, 1)))
        self.model.add(LSTM(units=60, return_sequences=False))
        self.model.add(Dense(units=30))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def prepare_data(self):
        data = self.df.values
        data_scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.numPastSamples, len(data_scaled) - self.numfutureSamples):
            X.append(data_scaled[i - self.numPastSamples:i, self.indexAttribute])
            y.append(data_scaled[i + self.numfutureSamples, self.indexAttribute])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def train(self):
        X_train, y_train = self.prepare_data()
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_trained_model(self):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict(self, df_future):
        if not self.load_trained_model():
            self.train()
        
        data_future = df_future.values
        data_future_scaled = self.scaler.transform(data_future)
        X_future = []
        for i in range(self.numPastSamples, len(data_future_scaled)):
            X_future.append(data_future_scaled[i - self.numPastSamples:i, self.indexAttribute])
        X_future = np.array(X_future)
        X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
        pred_future_scaled = self.model.predict(X_future)
        pred_future = self.scaler.inverse_transform(np.hstack((np.zeros_like(data_future_scaled), pred_future_scaled)))
        return pred_future[:, self.indexAttribute]

    def getPrediction(self, df_future):
        return self.predict(df_future)


# NeuronalNetV10Model
class NeuronalNetV10Model:
    def __init__(self, df, security):  
        
        params = security["params"]
        self.params = params
        self.period = params['period']
        self.df = df
        self.numPastSamples = params['numPastSamples']
        self.numfutureSamples = params['numfutureSamples']
        self.indexAttribute = params.get('indexAttribute', 3)
        self.epochs = params.get('epochs', 100)
        self.batch_size = params.get('batch_size', 256)
        self.model_path = params.get('model_path', 'model_v10.h5')
        self.scaler_path = params.get('scaler_path', 'scaler_v10.pkl')
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.model = None
        self.build_model()

    def build_model(self):
        self.model = Sequential()
        self.model.add(LSTM(units=80, return_sequences=True, input_shape=(self.numPastSamples, 1)))
        self.model.add(LSTM(units=80, return_sequences=False))
        self.model.add(Dense(units=40))
        self.model.add(Dense(units=1))
        self.model.compile(optimizer='adam', loss='mean_squared_error')

    def prepare_data(self):
        
        data = self.df.values
        data_scaled = self.scaler.fit_transform(data)
        X, y = [], []
        for i in range(self.numPastSamples, len(data_scaled) - self.numfutureSamples):
            X.append(data_scaled[i - self.numPastSamples:i, self.indexAttribute])
            y.append(data_scaled[i + self.numfutureSamples, self.indexAttribute])
        X, y = np.array(X), np.array(y)
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        return X, y

    def train(self):
        X_train, y_train = self.prepare_data()
        self.model.fit(X_train, y_train, batch_size=self.batch_size, epochs=self.epochs)
        self.model.save(self.model_path)
        joblib.dump(self.scaler, self.scaler_path)

    def load_trained_model(self, security):
        if os.path.exists(self.model_path) and os.path.exists(self.scaler_path):
            self.model = tf.keras.models.load_model(self.model_path)
            self.scaler = joblib.load(self.scaler_path)
            return True
        return False

    def predict(self, df_future):
        if not self.load_trained_model():
            self.train()
        
        data_future = df_future.values
        data_future_scaled = self.scaler.transform(data_future)
        X_future = []
        for i in range(self.numPastSamples, len(data_future_scaled)):
            X_future.append(data_future_scaled[i - self.numPastSamples:i, self.indexAttribute])
        X_future = np.array(X_future)
        X_future = np.reshape(X_future, (X_future.shape[0], X_future.shape[1], 1))
        pred_future_scaled = self.model.predict(X_future)
        pred_future = self.scaler.inverse_transform(np.hstack((np.zeros_like(data_future_scaled), pred_future_scaled)))
        return pred_future[:, self.indexAttribute]

    def getPrediction(self, df_future):
        return self.predict(df_future)



class PeaksAndValleysModel:
    
    def __init__(self, df, security):  
        
        self.security = security
        params = security["params"]
        self.bestDistanceValley = 8
        self.bestDistancePeak = 8
        self.bestPositionMargin = 10
        self.id = 'peaks_and_valleys'
        self.model = None
        self.period = params['period']
        self.params = params        
        samples = str(params['minNumPastSamples'])  + 'Samples'  

        self.TargetHours = range(7, 23)
        tb = self.TargetHours[0]
        te = self.TargetHours[-1]
        interval = str(tb) + '_' + str(te) + 'Hour'        
        nameParts = [ __name__, self.period, samples, interval]        
        self.fileName = '_'.join(nameParts)


    def isSuitableForThisTime(self, hour):
        return hour in self.TargetHours


    def findPeaksValleys(self, dataframe, sec, p):
        numWindowSize = 20
        if p == '1Min':
            numWindowSize = 100
        elif p == '30Min':
            numWindowSize = 160
        else:
            print('be careful')
            numWindowSize = 100
    
        # Ensure dataframe is not empty
        if dataframe.empty:
            print(f"DataFrame is empty for {sec} and period {p}")
            return {}
    
        dataframe = dataframe.tail(numWindowSize)
        
        # If tail returns empty, return early
        if dataframe.empty:
            print(f"DataFrame after tail is empty for {sec} and period {p}")
            return {}
    
        fluctuation = {}
        df = dataframe.copy()
        df['calcdatetime'] = dataframe.index
        
        #TODO temporarily
        df['calcdatetime'] = df['calcdatetime'].dt.tz_convert('America/New_York')
        #print(df['calcdatetime'])
        #print(df.index)
        
        fluctuation['samplingwindow'] = df[['calcdatetime','startprice','maxprice','minprice','endprice']]
    
        seriesAvg = (df['endprice'] + df['maxprice'] + df['minprice'] + df['startprice']) / 4
    
        # Ensure seriesAvg is not empty
        if seriesAvg.empty:
            print(f"seriesAvg is empty for {sec} and period {p}")
            return {}
    
        times = df['calcdatetime']
    
        b, a = signal.butter(2, 0.2)
        zi = signal.lfilter_zi(b, a)
    
        filtered, _ = signal.lfilter(b, a, seriesAvg, zi=zi*seriesAvg.iloc[0])
        filtered2, _ = signal.lfilter(b, a, filtered, zi=zi*filtered[0])
        y = signal.filtfilt(b, a, seriesAvg)
    
        #print('from ' + str(times.iloc[0]) + ' to ' + str(times.iloc[-1]))
    
        peak_idx_filtered, _ = find_peaks(y, distance=self.bestDistancePeak)
        valley_idx_filtered, _ = find_peaks(-y, distance=self.bestDistanceValley)
    
        fluctuation['peak_idx_filtered'] = peak_idx_filtered
        fluctuation['valley_idx_filtered'] = valley_idx_filtered
    
        peak_idx, _ = find_peaks(seriesAvg, distance=self.bestDistancePeak)
        valley_idx, _ = find_peaks(-seriesAvg, distance=self.bestDistanceValley)
    
        fluctuation['peak_idx'] = peak_idx
        fluctuation['valley_idx'] = valley_idx
    
        # Commented out AMPD-related code for now
        peaksAMPD = ampd.find_peaks(seriesAvg, numWindowSize)
        valleysAMPD = ampd.find_peaks(-seriesAvg, numWindowSize)
        plt.plot(seriesAvg, 'b')
        plt.plot(seriesAvg[peaksAMPD], 'k^', markersize=5)
        plt.plot(seriesAvg[valleysAMPD], 'rv', markersize=5)
        plt.title('Prediction for ' + p)
        plt.show()
    
        return fluctuation
    

    def plotPeaksAndValleys(self, seriesMax, seriesEnd, seriesMin, filteredDataLine, peak_idx, valley_idx, fluctuation, sec, times, p):
        seseccode = sec['seccode']
        label = sec.get('label', 'Unknown')

        if seseccode == 'GZM1':
            label = 'GAZPROM'
        elif seseccode == 'SRM1':
            label = 'SBERBANK'
        elif seseccode == 'SiM1':
            label = 'SIH'

        t = times
        plt.plot(t, seriesMax)
        plt.plot(t, seriesEnd)
        plt.plot(t, seriesMin)
        plt.plot(t, filteredDataLine, 'b')

        plt.plot(t[peak_idx], seriesMax[peak_idx], 'k^', markersize=15)
        plt.plot(t[valley_idx], seriesMin[valley_idx], 'rv', markersize=15)

        new_peak_ind = np.array(peak_idx) + 1
        plt.plot(t[new_peak_ind], seriesEnd[new_peak_ind], 'm^') 
        new_valley_ind = np.array(valley_idx) + 1
        plt.plot(t[new_valley_ind], seriesEnd[new_valley_ind], 'm^') 
        plt.title('Prediction for ' + label + '   ' + p)

        plt.show()

    
    def reviewForHigherfrequency (self, statusLowFreq, sec ):
        
        #FIXME
        periodHighFreq = self.periods[0]
        periodLowFreq = self.periods[-1]
        numPeriodLowFreq    =   int(periodLowFreq[0])
        numPeriodHighFreq   =   int(periodHighFreq[0])
        
        #FIXME
        periodGap = numPeriodLowFreq - numPeriodHighFreq  
        predictions = copy.deepcopy(sec['predictions'][periodHighFreq])                
        fluctuation = predictions[-1]
        numWindowSize = fluctuation['samplingwindow'].shape[0]
        indexPenultimate = numWindowSize - 2
        status = 0
      
        indexLastPeak = fluctuation['peak_idx'][-1] if fluctuation['peak_idx'].any() else 0
        indexLastValley = fluctuation['valley_idx'][-1] if fluctuation['valley_idx'].any() else 0
            
        if statusLowFreq == 1 and indexLastPeak > indexLastValley and \
            abs( indexPenultimate - indexLastPeak ) <= periodGap :            
            status = 1
        elif statusLowFreq == -1 and indexLastValley > indexLastPeak and \
            abs( indexPenultimate - indexLastValley ) <= periodGap:
            status = -1
        else: # statusLowFreq == 0
            status = 0 
        
        return status


    def predict( self, df, sec, p):       
        
        #FIXME bada bada bda
        # status = 'no-go' 
        # return status
        
        fluctuation = self.findPeaksValleys(df, sec, p)        
        numWindowSize = fluctuation['samplingwindow'].shape[0]  
        indexPenultimate = numWindowSize - 2
        indexLastPeak = fluctuation['peak_idx'][-1] if fluctuation['peak_idx'].any() else 0
        indexLastValley = fluctuation['valley_idx'][-1] if fluctuation['valley_idx'].any() else 0
   
        if indexLastPeak == indexPenultimate and indexLastPeak != indexLastValley :
            status = 'long'
        elif indexLastValley == indexPenultimate and indexLastPeak != indexLastValley :
             status = 'short'
        elif indexLastPeak == indexLastValley: #if in the same point thera peak and valley
             status = 'no-go'     
        else:
             status = 'no-go' 
             
        #status = self.reviewForHigherfrequency(status, sec)             
        #FIXME test
        #status = 'long'
        return status

