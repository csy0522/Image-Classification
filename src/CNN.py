#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Oct 18 06:11:07 2019

@author: csy
"""

import numpy as np
from data_with_notebook.scripts.util_mnist_reader import load_mnist
import tensorflow as tf
from functools import partial

PATH = "data_with_notebook/data/fashion"     # Data Path

'''
Convolutional Neural Network with Keras
This class creates a convolutional Neural Network with that consists:
  input layaer, convolutional layers, pooling layers, falltened Neural Network Layers, and output layer
'''
##################################### Convolutional Neural Network with Keras #####################################
class CNN:

  def __init__(self, input_, convolutions_, poolings_, ann_hiddens_, output_):
    self.input_ = input_
    self.convolutions_ = convolutions_
    self.poolings_ = poolings_
    self.ann_hiddens_ = ann_hiddens_
    self.output_ = output_
    
    
  '''
  This function initialize each layers mentions above
  All the layers are saved sequentially in order into the sequential list
  '''
  def __construct_cnn_elements__(self):
    
    self.sequential = []
    
    ###### Constructing Layers######
    
    # Input Layer #
    in_ = tf.keras.layers.Conv2D(
        filters = self.input_.filters_,
        kernel_size = self.input_.kernel_size_,
        strides = self.input_.strides_,
        padding = self.input_.padding_,
        activation= self.input_.activation_,
        input_shape=[28,28,1])
    
    # Convolutional Layers #
    convs_ = []
    for c in self.convolutions_:
      if type(c) == list:     # If the element is a list
        conv = []
        for cc in c:
          conv.append(tf.keras.layers.Conv2D(
            filters = cc.filters_,
            kernel_size = cc.kernel_size_,
            strides = cc.strides_,
            padding = cc.padding_,
            activation= cc.activation_))
        convs_.append(conv)
      else:
        convs_.append(tf.keras.layers.Conv2D(
          filters = c.filters_,
          kernel_size = c.kernel_size_,
          strides = c.strides_,
          padding = c.padding_,
          activation= c.activation_))
    
    # Pooling Layers
    pools_ = []
    for p in self.poolings_:
      if p.pooling_type_ == 'max':      # If the pooling type is max pooling
        pools_.append(
            tf.keras.layers.MaxPooling2D(pool_size=p.pooling_size_))
      elif p.pooling_type_ == 'avg':    # if the pooling type is average pooling
        pools_.append(
            tf.keras.layers.AveragePooling2D(pool_size=p.pooling_size_))
    
    # Flattened Neural Network Layers
    hids_ = []
    for h in self.ann_hiddens_:
      hids_.append(
          tf.keras.layers.Dense(units=h.units_, activation=h.activation_))
          
    # Output Layer
    out_ = tf.keras.layers.Dense(
        units=self.output_.units_,
        activation=self.output_.activation_)
    
    
    ###### Add the construction to sequence ######
    
    self.sequential.append(in_)
  
  
    '''
    The middle layers contains
    Pooling layers and convolutional layers 
    which are saved sequentially
    '''
    for i in range(max(len(convs_),len(pools_))):
      
      if i < len(pools_):
        if type(pools_[i]) == list:
          for p in pools_[i]:
            self.sequential.append(p)
        else:
          self.sequential.append(pools_[i])
      if i < len(convs_):
        if type(convs_[i]) == list:
          for c in convs_[i]:
            self.sequential.append(c)
        else:
          self.sequential.append(convs_[i])
      
          
    self.sequential.append(tf.keras.layers.Flatten())
      
    for c in hids_:
      self.sequential.append(c)
      self.sequential.append(tf.keras.layers.Dropout(0.5))
      
    self.sequential.append(out_)
  

  '''
  Building the CNN model using the sequential layers 
  saved from above
  '''
  def __build_cnn_model__(self):
    self.__construct_cnn_elements__()
    self.model_ = tf.keras.models.Sequential(self.sequential)
    self.model_.compile(
        loss = 'sparse_categorical_crossentropy',
        optimizer = 'sgd',
        metrics = ['accuracy'])

    
  '''
  Training
  Hyper parameters are epoch and batch size
  More hyper parameters can be adjusted from below
  '''
  def __train__(self,X,y,val,epoch,batch):
    self.training_info_ = self.model_.fit(X,y,validation_data=val,epochs=epoch,batch_size=batch,shuffle=True) 
    
    
    
  def __test__(self,test_X,test_y):
    self.score_ = self.model_.evaluate(test_X,test_y,verbose=0)
    print("=========================== TEST ==============================")
    print("loss: {}".format(self.score_[0]))
    print("accuracy: {}".format(self.score_[1]))
    print("===============================================================\n")


'''
Parameter Classes
These are designed to save the parameters for the layers
'''
###### Parameter Classes ######    
class Conv2d_Parameters:
  def __init__(self, filters_, kernel_size_, strides_, padding_, activation_):
    self.filters_ = filters_
    self.kernel_size_ = kernel_size_
    self.strides_ = strides_
    self.padding_ = padding_
    self.activation_ = activation_

class Pooling_Parameters:
  def __init__(self, pooling_type_, pooling_size_):
    self.pooling_type_ = pooling_type_
    self.pooling_size_ = pooling_size_
    
    
class Dense_Parameters:
  def __init__(self, units_, activation_):
    self.units_ = units_
    self.activation_ = activation_    
###### Parameter Classes ######
    
##################################### Convolutional Neural Network with Keras #####################################
    
    
    
    
    
'''
Helper Functions
These are used to assist model training
'''
##################################### Helper Functions #####################################
def get_data(path, kind, norm_size):
  X,y = load_mnist(path, kind)
  return X/norm_size,y


def get_validation_data(X,y,validation_percentage):
  validation_X = []
  validation_y = []
  random_index = np.random.permutation(len(X))
  val_size = int(len(X) * validation_percentage)
  for i in range(val_size):
    validation_X.append(X[random_index[i]])
    validation_y.append(y[random_index[i]])
  return np.array(validation_X), np.array(validation_y)

'''
This function initiazes the default layer settings for each type of layers
So everytime a new layer is created there will not be repetitive coding
'''
def get_partial():
  def_conv2d = partial(Conv2d_Parameters,filters_=3,kernel_size_=7,strides_=(1,1),padding_='SAME',activation_='relu')
  def_pool = partial(Pooling_Parameters,pooling_type_='max', pooling_size_=2)
  def_dense = partial(Dense_Parameters,units_=128, activation_='relu')
  return def_conv2d, def_pool, def_dense

'''
Parameters for the CNN model
These parameters can be adjusted for a better model
'''
def get_cnn_params():
  def_conv2d, def_pool, def_dense = get_partial()
  cnn_input = def_conv2d()
  cnn_convs = [def_conv2d(filters_=256)]
  cnn_pools = [def_pool()]
  cnn_hids = [def_dense(units_=64)]
  cnn_output = def_dense(units_=10,activation_='softmax')
  return cnn_input, cnn_convs, cnn_pools, cnn_hids, cnn_output
##################################### Helper Functions #####################################




#if __name__ == '__main__':
#  
#  ### Hyper Parameters used to adjust the result ###
#  ### More Hyper parameters can be adjusted from 'get_partial' function on line 171 ###
#  epoch = 6
#  batch_size = 100
#  ### The parameter classes for CNN class ###
#  cnn_input, cnn_convs, cnn_pools, cnn_hids, cnn_output = get_cnn_params()
#  
#  
#  ### Data ###
#  X,y = get_data(PATH, kind='train', norm_size=255)
#  np.random.shuffle(X);np.random.shuffle(y)
#  val_X,val_y = get_validation_data(X,y,0.2)      ### use 20% of the training data for validation ###
#  X = np.reshape(X,(X.shape[0],28,28,1))          ### Reshape the flattened image to 3D ###
#  val_X = np.reshape(val_X,(val_X.shape[0],28,28,1))
#  test_X,test_y = get_data(PATH, kind='t10k', norm_size=255)
#  test_X = np.reshape(test_X,(test_X.shape[0],28,28,1))
#  
#  ### Convolutional Neural Network ###
#  cnn = CNN(cnn_input, cnn_convs, cnn_pools, cnn_hids, cnn_output)
#  cnn.__build_cnn_model__()
#  cnn.__train__(X,y,(val_X,val_y),epoch,batch_size)
#  cnn.__test__(test_X,test_y)
    
    
    
    

    
    
    
    
    
    
    
    