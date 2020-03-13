#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 04:56:25 2019

@author: csy
"""

import numpy as np
from Data_Reader import load_mnist
import tensorflow as tf
import matplotlib.pyplot as plt

PATH = "../data"        # Data Path
      
'''
Multilayers Neural Network with Keras
Cross Entropy is used for the loss function,
Stochasitc Gradient Descent is used for the optimizer
Accuracy is used to measure the model accuracy
'''     
##################################### Multilayers Neural Network with Keras #####################################
class MLANN:
  
    def __init__(self, input_, hiddens_, output_, activation_):
        self.input_ = input_
        self.hiddens_ = hiddens_
        self.output_ = output_
        self.activation_ = activation_
      
      
    '''
    Building Artifitial Neural Network using Keras
    The number of hidden layers is one of the hyper parameters the user can adjust
    '''
    def __build_keras_ann__(self):
        self.model_ = tf.keras.models.Sequential()
        self.model_.add(tf.keras.layers.Flatten(input_shape=[self.input_]))
        for h in self.hiddens_:
            self.model_.add(tf.keras.layers.Dense(h, activation=self.activation_))
        self.model_.add(tf.keras.layers.Dense(self.output_, activation='softmax'))
        self.model_.compile(
            loss = 'sparse_categorical_crossentropy',
            optimizer = 'sgd',
            metrics = ['accuracy'])
      
     
    '''
    Training
    epochs & batch_size can be adjusted
    '''
    def __train__(self,X,y,val,epoch,batch):
        self.training_info_ = self.model_.fit(X,y,validation_data=val,epochs=epoch,batch_size=batch,shuffle=True)
        self.__plot_training_progress__()
      
      
    def __test__(self,test_X,test_y):
        self.score_ = self.model_.evaluate(test_X,test_y,verbose=0)
        print("=========================== TEST ==============================")
        print("loss: {}".format(self.score_[0]))
        print("accuracy: {}".format(self.score_[1]))
        print("===============================================================\n")
    
    
    
    def __plot_training_progress__(self):
        plt.figure(figsize=(10,8))
        plt.title("Accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("Accuracy")
        plt.plot(self.training_info_.history["accuracy"])
        plt.legend()
        plt.show()
##################################### Multilayers Neural Network with Keras #####################################
    
  







'''
Helper Functions
The following are used for assisting training
'''
##################################### Helper Functions #####################################
def get_data(path, kind, norm_size):
    X,y = load_mnist(path, kind)
    return X/norm_size,y


def get_validation_data(X,y,percent):
    validation_X = []
    validation_y = []
    random_index = np.random.permutation(len(X))
    val_size = int(len(X) * percent)
    for i in range(val_size):
        validation_X.append(X[random_index[i]])
        validation_y.append(y[random_index[i]])
    return np.mat(validation_X), np.array(validation_y)
##################################### Helper Functions #####################################



  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  