#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 04:56:25 2019

@author: csy
"""

import numpy as np
from data_with_notebook.scripts.util_mnist_reader import load_mnist
import tensorflow as tf

PATH = "data_with_notebook/data/fashion"     # Data Path
      
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
    
    
    
  def __test__(self,test_X,test_y):
    self.score_ = self.model_.evaluate(test_X,test_y,verbose=0)
    print("=========================== TEST ==============================")
    print("loss: {}".format(self.score_[0]))
    print("accuracy: {}".format(self.score_[1]))
    print("===============================================================\n")
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









#if __name__ == '__main__':
#  
#  ### Hyper Parameters used to adjust the result ###
#  hiddens = [300,50]
#  activation = 'relu'
#  epoch = 2
#  batch_size = 100
#
#  ### Data ###
#  X,y = get_data(PATH, kind='train', norm_size=255)
#  val_X,val_y = get_validation_data(X,y,0.2)      ### use 20% of the training data for validation ###
#  test_X,test_y = get_data(PATH, kind='t10k', norm_size=255)
#  
#  ### Neural Network Model with Keras ###
#  mlann = MLANN(X.shape[1],hiddens,10,activation)
#  mlann.__build_keras_ann__()
#  mlann.__train__(X,y,(val_X,val_y),epoch,batch_size)
#  mlann.__test__(test_X, test_y)
  



  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  