#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 14 04:56:25 2019

@author: csy
"""

import numpy as np
from data_with_notebook.scripts.util_mnist_reader import load_mnist

PATH = "data_with_notebook/data/fashion"     # Data Path

'''
Artifitial Neural Network

This is a Neural Network with 1 hidden layer
Inputs are 60000 images with 784 pixels
Output has 10 categories
The prediction is based on the percentage of predictions
The highest percentages among all the prediction is the predicted answer.
'''
##################################### Artifitial Neural Network #####################################
class ANN:
  
  '''
  Initialize node numbers for input, hidden, and output
  The list of activation functions are sigmoid, tanh, and relu
  '''
  def __init__(self, input_, hiddens_, output_, activation_):
    self.input_ = input_
    self.hiddens_ = hiddens_
    self.output_ = output_
    if activation_ == 'sigmoid':
      self.activation_ = sigmoid
      self.activation_grad_ = sigmoid_grad
    elif activation_ == 'relu':
      self.activation_ = relu
      self.activation_grad_ = relu_grad
    elif activation_ == 'tanh':
      self.activation_ = tanh
      self.activation_grad_ = tanh_grad
    
    
  '''
  Initialize Weights and Biases
  Weights range from -1 to 1
  Biases are 0s
  '''
  def __initialize_weights_biases__(self):
    self.weights_ = {}
    self.biases_ = {}
    current_input = self.input_
    for i in range(len(self.hiddens_)):
      self.weights_[i] = np.random.uniform(-1,1,(current_input, self.hiddens_[i]))
      self.biases_[i] = np.zeros(self.hiddens_[i])
      current_input = self.hiddens_[i]
    self.weights_[len(self.hiddens_)] = np.random.uniform(-1,1,(current_input, self.output_))
    self.biases_[len(self.hiddens_)] = np.zeros(self.output_)
    
    
  '''
  Initialize Hyper Parameters
  Hyper parameters are learning rate, learning rate decreasing rate, epoch, and batch size
  These hyper parameters will be adjusted to get the best training result
  '''
  def __initialize_hyperparameters__(self,learning_rate,learning_rate_decrease,epoch,batch):
    self.learning_rate_ = learning_rate
    self.learning_rate_decrease_ = learning_rate_decrease
    self.epoch_ = epoch
    self.batch_size_ = batch
    
    
  '''
  Forward Propagation
  Z - Dictionary that contains results [sum(weights * input) + biases] for the hidden and output layer
  A - Dictionary that contains Z after activation function is applied
  '''
  def __feed_forward__(self, X):
    self.z_ = {}
    self.a_ = {}
    current_input = X
    self.z_[0] = current_input
    for i in range(len(self.hiddens_)):
      self.z_[i+1] = np.dot(current_input, self.weights_[i]) + self.biases_[i]
      current_input = self.activation_(self.z_[i+1])
      self.a_[i+1] = current_input
    self.z_[len(self.hiddens_)+1] = np.dot(current_input, self.weights_[len(self.hiddens_)]) + self.biases_[len(self.hiddens_)]
    self.a_[len(self.hiddens_)+1] = softmax(self.z_[len(self.hiddens_)+1])


  '''
  Back Propagation
  This step calculates the amount of errors and back propagate to the beginning and update the parameters
  Chain rule is used for the weights & biases gradients
  @L/@W = @L/@a x @a/@z x @z/@w
  @L/@b = @L/@y x @a/@z x @z/@b
  '''
  def __back_propagation__(self,X,y):
    y_hot = get_one_hot(y,self.output_)
    loss = self.a_[2] - y_hot 
#    j2 = np.dot(self.a_[1].T, (softmax_grad(self.a_[2]) * loss))
    j2 = np.dot(self.a_[1].T, loss)
#    j1 = np.dot(X.T, (np.dot((softmax_grad(self.a_[2]) * loss), self.weights_[1].T) * self.activation_grad_(self.a_[1])))
    j1 = np.dot(X.T, (np.dot(loss, self.weights_[1].T)))
    self.biases_[1] - (self.learning_rate_ * np.sum(loss, axis=0))
    self.biases_[0] - (self.learning_rate_ * np.sum(np.dot((softmax_grad(self.a_[2]) * loss), self.weights_[1].T), axis=0))
    self.weights_[1] -= self.learning_rate_ * j2
    self.weights_[0] -= self.learning_rate_ * j1

  
  '''
  Training
  For every epoch, do
    for every batch sized data, data
      forward propagate - calculate error - back propagate
  '''
  def __train__(self,X,y):
    self.accuracy_ = []
    init_index = 0
    for e in range(self.epoch_):
      for i in range(int(len(X)/self.batch_size_)):
        batch_X = X[init_index:init_index + self.batch_size_]
        batch_y = y[init_index:init_index + self.batch_size_]
        self.__feed_forward__(batch_X)
        self.__back_propagation__(batch_X,batch_y)
        print("===============================================================")
        print("Epoch: {}/{}".format(e+1,self.epoch_))
        print("Iteration: {}/{}".format(init_index+i+1,len(X)))
        print("loss: {}".format(cross_entropy(batch_y,self.a_[2])))
        print("accuracy: {}".format(accuracy(batch_y,self.a_[2])))
        self.accuracy_.append(accuracy(batch_y,self.a_[2]))
        print("===============================================================\n")
        self.learning_rate_ = self.learning_rate_ * self.learning_rate_decrease_
        init_index += self.batch_size_
        if init_index >= len(y):
          init_index = 0
          
          
  def __test__(self,test_X,test_y):
    self.__feed_forward__(test_X)
    print("=========================== TEST ==============================")
    print("loss: {}".format(cross_entropy(test_y,self.a_[2])))
    print("accuracy: {}".format(accuracy(test_y,self.a_[2])))
    self.accuracy_.append(accuracy(test_y,self.a_[2]))
    print("===============================================================\n")
##################################### Artifitial Neural Network #####################################
    
  
  
  
  
  
'''
Activation & Activation Gradient Functions
The following are the activation options for the training,
which includes sigmoid, tanh, relu, and softmax
softmax is used at the end layer
'''
##################################### Activation & Activation Gradient #####################################
def sigmoid(X):
  return 1/(1+np.exp(-X))
  

def sigmoid_grad(X):
  sig = sigmoid(X)
  return sig/(1 - sig)


def relu(X):
  return np.where(X <= 0, 0, X)


def relu_grad(X):
  return np.where(X > 0, 1, 0)

def tanh(X):
  return (np.exp(X) - np.exp(-X)) / (np.exp(X) + np.exp(-X))

def tanh_grad(X):
  return 1-(tanh(X)**2)


def softmax(X):
  XX = np.zeros(X.shape)
  if len(X.shape) == 2:   # if the parameter X is 2 dimension
    for i in range(X.shape[0]):
      e = np.exp(X[i] - np.max(X[i]))
      es = np.sum(e)
      XX[i] = e / es
    return XX
  else:                   # if the parameter X is 1 dimension
    e = np.exp(X - np.max(X))
    es = np.sum(e)
    return e / es
  

def softmax_grad(X):
  soft_max = softmax(X)
  return soft_max/(1-soft_max)
##################################### Activation & Activation Gradient #####################################
  






'''
Helper Functions
The following are used to assist the training process
'''
##################################### Helper Functions #####################################
def get_data(path, kind, norm_size):
  X,y = load_mnist(path, kind)
  return X/norm_size,y


def get_validation_data(X,y,validation_percent):
  validation_X = []
  validation_y = []
  random_index = np.random.permutation(len(X))
  num_val = int(len(X) * validation_percent)
  for i in range(num_val):
    validation_X.append(X[random_index[i]])
    validation_y.append(y[random_index[i]])
  return np.mat(validation_X), np.array(validation_y)


def get_mini_batches(X,y,batch_size):
  X_batches = {}
  y_batches = {}
  num_batches = int(len(X) / batch_size)
  init_batch = 0
  for i in range(num_batches):
    X_batches[i] = np.mat(X[init_batch:init_batch+batch_size])
    y_batches[i] = np.array(y[init_batch:init_batch+batch_size])
    init_batch += batch_size
  return X_batches, y_batches


def get_one_hot(y,column):
  one_hot = np.zeros((y.shape[0], column))
  if len(y.shape) == 2:
    y = one_hot_reverse(y)
  for i in range(len(y)):
    one_hot[i][y[i]] = 1
  return one_hot
      

def one_hot_reverse(one_hot):
  return np.argmax(one_hot, axis=1)


def cross_entropy(y,a):
  y_hot = get_one_hot(y,a.shape[1])
  loss = -(1/len(y)) * np.sum(y_hot * np.log(a+0.000001))
  return loss


def accuracy(y,pred):
  pred = one_hot_reverse(pred)
  correct = 0
  for i in range(y.shape[0]):
    if y[i] == pred[i]:
      correct += 1
  return correct / y.shape[0]
##################################### Helper Functions #####################################




if __name__ == '__main__':
  
  ### Hyper Parameters used to adjust the result ###
  hidden = [100]
  activation = 'sigmoid'
  learning_rate = 0.0001
  learning_rate_descent = 1
  epoch = 10
  batch_size = 6000
  
  ### Data ###
  data_kind = 'train'
  X,y = get_data(PATH, kind=data_kind, norm_size=255)
  val_X,val_y = get_validation_data(X,y,0.2)      ### use 20% of the training data for validation ###
  test_X,test_y = get_data(PATH, kind='t10k', norm_size=255)
  
  
  ### Neural Network Model ###
  ann = ANN(X.shape[1],hidden,10,activation)
  ann.__initialize_weights_biases__()     
  ann.__initialize_hyperparameters__(
      learning_rate,
      learning_rate_descent,
      epoch,
      batch_size)   
  ann.__train__(X,y)
  ann.__test__(test_X,test_y)
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  
  