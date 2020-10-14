# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 21:43:12 2020

@author: mvereda
"""

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler
# tf.compat.v1.disable_eager_execution()
import gc; gc.collect()
import pandas as pd
from datetime import timedelta
    
num_layers = 1
learning_rate = 0.005
size_layer = 128
forget_bias = 0.8
timestamp = 5
epoch = 500
dropout_rate = 0.6


class Model:
    def __init__( self,  df , params):
        
        self.dates = df.index.tolist()
        self.loadData(df)
        self.numAttributes = self.df_scaled.shape[1]
        self.epoch = epoch
        self.num_layers = num_layers
        self.size_layer = size_layer 
        self.forget_bias = forget_bias
        self.learning_rate = learning_rate
        
        def lstm_cell(size_layer):
            return tf.nn.rnn_cell.LSTMCell(size_layer, state_is_tuple = False)

        rnn_cells = tf.nn.rnn_cell.MultiRNNCell(
            [lstm_cell(size_layer) for _ in range(num_layers)],
            state_is_tuple = False,
        )
        self.X = tf.placeholder(tf.float32, (None, None, self.numAttributes))
        self.Y = tf.placeholder(tf.float32, (None, self.numAttributes))
        drop = tf.contrib.rnn.DropoutWrapper( 
            rnn_cells,
            output_keep_prob = forget_bias
        )
        self.hidden_layer = tf.placeholder( 
            tf.float32,
            (None, num_layers * 2 * size_layer)
        )
        self.outputs, self.last_state = tf.nn.dynamic_rnn(
            drop, 
            self.X, 
            initial_state = self.hidden_layer, 
            dtype = tf.float32
        )
        self.logits = tf.layers.dense(
            self.outputs[-1],
            self.numAttributes,
            kernel_initializer = tf.glorot_uniform_initializer(),
        )
        self.cost = tf.reduce_mean( tf.square(self.Y - self.logits) )
        
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
            self.cost
        )
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
        self.scaler = MinMaxScaler().fit( df[listAttributes].astype('float32' ) )        
       
        df_scaled = self.scaler.transform( df[listAttributes].astype('float32' ) )
        self.df_scaled = pd.DataFrame(df_scaled)
        
        
    def train( _ ):        
  
        numSamples = _.df_scaled.shape[0]
        ts = _.timestamp
        
        for i in range(_.epoch):
            init_value = np.zeros((1, _.num_layers * 2 * _.size_layer))
            total_loss = 0
            for k in range(0, (numSamples // ts) * ts, ts):
                batch_x = np.expand_dims(
                    _.df_scaled.iloc[k : k + ts].values, axis = 0
                )   
                batch_y = _.df_scaled.iloc[k + 1 : k + ts + 1].values
                last_state, x, loss = _.sess.run(
                    [_.last_state, _.optimizer, _.cost],
                    feed_dict = {
                        _.X: batch_x,
                        _.Y: batch_y,
                        _.hidden_layer: init_value,
                    },
                )
                init_value = last_state
                total_loss += loss
            total_loss /= numSamples // ts
            if (i + 1) % 2 == 0:
                print('epoch:', i + 1, 'avg loss:', total_loss)   

    def predict( _ , future_count, indices = {}):
        
        ts = _.timestamp
        date_ori = _.dates[:]
        cp_df = _.df.copy()
        #numRows = cp_df.shape[0]
        numCols = cp_df.shape[1]

        output_predict = np.zeros((cp_df.shape[0] + future_count, numCols))
        output_predict[0] = cp_df.iloc[0]
        upper_b = (cp_df.shape[0] // ts) * ts
        init_value = np.zeros((1, _.num_layers * 2 * _.size_layer))
        for k in range(0, (cp_df.shape[0] // ts) * ts, ts):
            out_logits, last_state = _.sess.run(
                [_.logits, _.last_state],
                feed_dict = {
                    _.X: np.expand_dims(
                        cp_df.iloc[k : k + ts], axis = 0
                    ),
                    _.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[k + 1 : k + ts + 1] = out_logits
        out_logits, last_state = _.sess.run(
            [_.logits, _.last_state],
            feed_dict = {
                _.X: np.expand_dims(cp_df.iloc[upper_b:], axis = 0),
                _.hidden_layer: init_value,
            },
        )
        init_value = last_state
        output_predict[upper_b + 1 : cp_df.shape[0] + 1] = out_logits
        cp_df.loc[cp_df.shape[0]] = out_logits[-1]
        date_ori.append(date_ori[-1] + timedelta(minutes = 1))
        if indices:
            for key, item in indices.items():
                cp_df.iloc[-1, key] = item
        for i in range(future_count - 1):
            out_logits, last_state = _.sess.run(
                [_.logits, _.last_state],
                feed_dict = {
                    _.X: np.expand_dims(cp_df.iloc[-ts:], axis = 0),
                    _.hidden_layer: init_value,
                },
            )
            init_value = last_state
            output_predict[cp_df.shape[0]] = out_logits[-1]
            cp_df.loc[cp_df.shape[0]] = out_logits[-1]
            date_ori.append(date_ori[-1] + timedelta(minutes = 1))
            if indices:
                for key, item in indices.items():
                    cp_df.iloc[-1, key] = item
                    
        preds = _.scaler.inverse_transform( cp_df.values )
        
        return { 
                'date_ori': date_ori, 
                'df': preds,
                'algorithm':  __name__ 
                }
    