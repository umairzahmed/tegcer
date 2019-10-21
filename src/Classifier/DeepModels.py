'''Defines various Deep-Network layering, their train-test and loss functions. Any of the DeepModel can be invoked based on its "modelName"'''

from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed, Dropout
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.layers.embeddings import Embedding
from keras import backend as K
import tensorflow as tf
import keras
from keras.backend.tensorflow_backend import set_session

from collections import defaultdict
from termcolor import colored
import random, math, os, sys, datetime
import numpy as np

from src.Base import ConfigFile as CF, Helper as H
from DataEncoding import *

config = tf.ConfigProto()
#config.gpu_options.allow_growth=True
set_session(tf.Session(config=config))

def customLoss_binary_crossentropy(y_true, y_pred):
    '''Custom binary cross-entropy loss function, which penalizes mis-prediction of 1-label, and ignore matching 0-label.
In other words, predicting correctly that a particular label is NOT valid for an example, doesn't contribute to the loss
Only predicting an incorrect label, or missing out on a label does. '''
    new_y_true, new_y_pred = [], []
    #for i, j in zip(tf.unstack(y_true), tf.unstack(y_pred)):
    #    if i!=0 and j!=0:
    #        new_y_true.append(i)
    #        new_y_pred.append(j)

    return K.mean(K.binary_crossentropy(y_true, y_pred), axis=-1)

class DeepModel:
    funcNameStart = 'addLayers_'

    def __init__(self, modelName, dataset):
        print colored('\tCreating deep model: ' + modelName + '...', 'magenta')

        self.model = Sequential()
        self.modelName = modelName
        self.dataset = dataset
        self.addFirstLayer()
        self.graph = tf.get_default_graph() # Have to do this to support web-service (using Flask)
    
    def __str__(self):
        return '\n'.join([str(layer.output.name) +' '+ str(layer.output.shape) for layer in self.model.layers])
    
    def addFirstLayer(self):
        self.dataset.setDefaultTrainSet(self.modelName)
        # Add the layers corressponding to the modelName
        funcAddLayer = getattr(self, DeepModel.funcNameStart + self.modelName)
        funcAddLayer()

    def addLastLayer(self):
        if self.dataset.multiClass: # If multi-label problem, use sigmoid with ??? as loss function
            self.model.add(Dense(self.dataset.num_labels, activation='sigmoid'))
            self.model.compile(loss=customLoss_binary_crossentropy, optimizer='adam', metrics=['accuracy']) 
            # loss = ['mean_squared_error', 'binary_crossentropy', 'customLoss_binary_crossentropy]

        else:    # Otherwise, use a softmax layer with categorical_crossentropy as loss function
            self.model.add(Dense(self.dataset.num_classes, activation='softmax'))
            self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy']) 

        print(self.model.summary())

    def train(self, epochs, train_mult_factor):
        self.addLastLayer()
        print colored('\tTraining deep model: ' + self.modelName + '...', 'magenta')
        self.trainX, self.trainY = self.dataset.X_train.tolist()* train_mult_factor, self.dataset.y_train.tolist()* train_mult_factor

        startTime = datetime.datetime.now()
        keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=1, verbose=1, mode='auto')
        history = self.model.fit(self.trainX, self.trainY, epochs=epochs, validation_data=(self.dataset.X_valid, self.dataset.y_valid), verbose=1)
        return history, datetime.datetime.now() - startTime

    def test(self):
        scores = self.model.evaluate(self.dataset.X_test, self.dataset.y_test, verbose=1)
        acc = scores[1]*100
        print('\n\n-- Accuracy: %.2f%% \n' % (acc))
        return round(acc, 2)

    def getPrediction(self, PREDICT_TOP_K=1):
        with self.graph.as_default(): # Have to do this to support web-service (using Flask)
            preds_topk = []
            predClassesTests = self.model.predict(self.dataset.X_test, verbose=1)

            for predClasses in predClassesTests: # For each test case
                indices = predClasses.argsort()[-PREDICT_TOP_K:][::-1] # Fetch top-k, desc
                preds_topk.append(indices)

            # print '--Orig Top K=1 Class--', predClasses_Tests
            # predClasses_Tests = self.model.predict_classes(self.dataset.X_test, verbose=1)

            return preds_topk
        

    # ----- LAYERS ----------
    def addLayers_experimental(self):
        self.dataset.X_train, self.dataset.X_valid, self.dataset.X_test = self.dataset.X_train_bin, self.dataset.X_valid_bin, self.dataset.X_test_bin
        
        self.model.add(Dense(512, input_shape=(self.dataset.max_vocab_size,), activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(Dense(512, input_shape=(self.dataset.max_vocab_size,), activation='relu'))
        self.model.add(Dropout(0.2))

    def addLayers_Dense(self):
        self.model.add(Dense(512, input_shape=(self.dataset.max_seq_length,), activation='relu'))

    def addLayers_Dense_Bin(self):        
        self.model.add(Dense(512, input_shape=(self.dataset.max_vocab_size,), activation='relu'))
        self.model.add(Dropout(0.2))

    def addLayers_LSTM(self):
        timesteps, data_dim = self.dataset.max_seq_length, 1 # expected input data shape: (batch_size, timesteps, data_dim) - (None, 58, 32)
        self.dataset.X_train = self.dataset.X_train.reshape(self.dataset.X_train.shape[0], self.dataset.X_train.shape[1], data_dim) # Convert 2D to 3D: At each timestep 1 dim value
        self.dataset.X_valid = self.dataset.X_valid.reshape(self.dataset.X_valid.shape[0], self.dataset.X_valid.shape[1], data_dim) # Convert 2D to 3D: At each timestep 1 dim value
        self.dataset.X_test = self.dataset.X_test.reshape(self.dataset.X_test.shape[0], self.dataset.X_test.shape[1], data_dim) # Convert 2D to 3D

        self.model.add(LSTM(32, return_sequences=True, input_shape=(timesteps, data_dim))) 
        self.model.add(LSTM(32, return_sequences=True)) 
        self.model.add(LSTM(32))  # return a single vector of dimension 32
        #self.model.add(Dropout(0.2))

    def todo_addLayers_LSTM_stateful(self): # todo
        self.model.add(Dense(512, input_shape=(self.dataset.max_seq_length,), activation='relu'))
        self.model.add(LSTM(100))

    def addLayers_embed_LSTM(self):
        self.model.add(Embedding(self.dataset.max_vocab_size, CF.EMBEDDING_VECTOR_LENGTH, input_length=self.dataset.max_seq_length))
        self.model.add(LSTM(32, return_sequences=True)) 
        self.model.add(LSTM(50))  # return a single vector of dimension 32

    def ignore_addLayers_embed_LSTM_Dropout(self):
        self.model.add(Embedding(self.dataset.max_vocab_size, CF.EMBEDDING_VECTOR_LENGTH, input_length=self.dataset.max_seq_length))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32, return_sequences=True)) 
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32, return_sequences=True)) 
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(32))  # return a single vector of dimension 32

    def ignore_addLayers_embed_LSTM_Dropout_Recur(self):
        self.model.add(Embedding(self.dataset.max_vocab_size, CF.EMBEDDING_VECTOR_LENGTH, input_length=self.dataset.max_seq_length))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, recurrent_dropout=0.2)) #, recurrent_dropout=0.2
        self.model.add(Dropout(0.2))

    def addLayers_embed_CNN_LSTM(self):
        self.model.add(Embedding(self.dataset.max_vocab_size, CF.EMBEDDING_VECTOR_LENGTH, input_length=self.dataset.max_seq_length))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(LSTM(100))

    def addLayers_embed_CNN_LSTM_Dropout(self):
        self.model.add(Embedding(self.dataset.max_vocab_size, CF.EMBEDDING_VECTOR_LENGTH, input_length=self.dataset.max_seq_length))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100))
        self.model.add(Dropout(0.2))

    def addLayers_embed_CNN_LSTM_LSTM_Dropout(self):
        self.model.add(Embedding(self.dataset.max_vocab_size, CF.EMBEDDING_VECTOR_LENGTH, input_length=self.dataset.max_seq_length))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(50))
        self.model.add(Dropout(0.2))

    def addLayers_embed_CNN_LSTM_Dropout_Recur(self):
        self.model.add(Embedding(self.dataset.max_vocab_size, CF.EMBEDDING_VECTOR_LENGTH, input_length=self.dataset.max_seq_length))
        self.model.add(Dropout(0.2))
        self.model.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
        self.model.add(Dropout(0.2))
        self.model.add(MaxPooling1D(pool_size=2))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100, recurrent_dropout=0.2))
        self.model.add(Dropout(0.2))

allModelNames = [func[len(DeepModel.funcNameStart):] 
    for func in dir(DeepModel) if func.startswith(DeepModel.funcNameStart)]

