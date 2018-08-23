import tensorflow as tf
import numpy as np 
import random
from collections import deque

from __future__ import print_function
import keras
from keras import models
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import matplotlib.pyplot as plt

import os

# Hyper Parameters:
FRAME_PER_ACTION = 1
GAMMA = 0.95 # decay rate of past observations
OBSERVE = 50000. # timesteps to observe before training
EXPLORE = 1000000. # frames over which to anneal epsilon
FINAL_EPSILON = 0.1#0.001 # final value of epsilon
INITIAL_EPSILON = 1.0#0.01 # starting value of epsilon
REPLAY_MEMORY = 40000 # number of previous transitions to remember
BATCH_SIZE = 32 # size of minibatch
UPDATE_TIME = 10000

class BrainDQN3L:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
		# init Q network
		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2,self.W_fc3,self.b_fc3 = self.createQNetwork()

		# init Target Q Network
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T,self.W_fc3T,self.b_fc3T = self.createQNetwork()

		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2),self.W_fc3T.assign(self.W_fc3),self.b_fc3T.assign(self.b_fc3)]

		self.createTrainingMethod()

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state("./savedweightsKeras")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
				print "Could not find old network weights"


	def createQNetwork(self):

		# network weights
		# W_conv1 = self.weight_variable([6,6,4,16])
		# b_conv1 = self.bias_variable([16])
        #
		# W_conv2 = self.weight_variable([4,4,16,32])
		# b_conv2 = self.bias_variable([32])
        #
		# W_conv3 = self.weight_variable([2,2,32,32])
		# b_conv3 = self.bias_variable([32])
        #
		# W_fc1 = self.weight_variable([1056,176])
		# b_fc1 = self.bias_variable([176])
        #
		# W_fc2 = self.weight_variable([176,44])
		# b_fc2 = self.bias_variable([44])
        #
		# W_fc3 = self.weight_variable([44,self.actions])
		# b_fc3 = self.bias_variable([self.actions])

		# Image dimensions :
		width = 65
		height = 160
		depth = 4

		# input layer
		stateInput = tf.placeholder("float",[None,width,height,depth])
		# stateInpu = keras.im

		# Keras tryout :
		model = Sequential()

		# first cnn layer
		model.add(Conv2D(16, (6, 6), activation='relu', padding='same', input_shape=(stateInput.shape[,:,:,:]),
                         name='h_conv1'))
		# Maxpooling 2x2
		model.add(MaxPooling2D(pool_size=(2, 2)))
		# second cnn layer
		model.add(Conv2D(32, (4, 4), activation='relu', name='h_conv2'))
		# third cnn layer
		model.add(Conv2D(32, (2, 2), activation='relu', name='h_conv3'))
		# adding dropout with probability .5
		model.add(Dropout(0.5))
		# vectorising the
		model.add(Flatten(), name='h_conv3_flat')
		# adding 3 fully connected layers
		model.add(Dense(176, activation='relu', kernel_regularizer='l2', name='h_fc1'))
		model.add(Dense(44, activation='relu', kernel_regularizer='l2', name='h_fc2''))
		model.add(Dense(len(self.actions), activation='relu', kernel_regularizer='l2', name='h_fc3'))

		model.compile()

        QValue = model.fit(stateInput)


		# # hidden layers
		# h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,3) + b_conv1)
		# # max pool 2x2
		# h_pool1 = self.max_pool_2x2(h_conv1)
        #
		# h_conv2 = tf.nn.relu(self.conv2d(h_pool1,W_conv2,2) + b_conv2)
        #
        #
		# h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
		# h_conv3_shape = h_conv3.get_shape().as_list()
		# print "dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3]
        #
		# h_conv3_shape = model.h_conv3.get_shape().as_list()
		# print "dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3]
        #
		# h_conv3_flat = tf.reshape(h_conv3,[-1,1056])
		# h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
        #
		# h_fc2 = tf.nn.relu(tf.matmul(h_fc1,W_fc2)+ b_fc2)

		# Retrieving the weights and biases of each layer
		W_conv1 = tf.cast(model.layers[0].get_weights()[0], tf.Variable)
		b_conv1 = tf.cast(model.layers[0].get_weights()[1], tf.Variable)

		W_conv2 = tf.cast(model.layers[1].get_weights()[0], tf.Variable)
		b_conv2 = tf.cast(model.layers[1].get_weights()[1], tf.Variable)

		W_conv3 = tf.cast(model.layers[2].get_weights()[0], tf.Variable)
		b_conv3 = tf.cast(model.layers[2].get_weights()[1], tf.Variable)

		W_fc1 = tf.cast(model.layers[3].get_weights()[0], tf.Variable)
		b_fc1 = tf.cast(model.layers[3].get_weights()[1], tf.Variable)

		W_fc2 = tf.cast(model.layers[4].get_weights()[0], tf.Variable)
		b_fc2 = tf.cast(model.layers[4].get_weights()[1], tf.Variable)

		W_fc3 = tf.cast(model.layers[5].get_weights()[0], tf.Variable)
		b_fc3 = tf.cast(model.layers[5].get_weights()[1], tf.Variable)



		# Q Value layer
		# QValue = tf.matmul(h_fc2,W_fc3) + b_fc3

		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2,W_fc3,b_fc3


	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)


	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
		self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		self.trainStep = tf.train.RMSPropOptimizer(0.00025,0.99,0.0,1e-6).minimize(self.cost)


	def trainQNetwork(self):

		# Step 1: obtain random minibatch from replay memory
		minibatch = random.sample(self.replayMemory,BATCH_SIZE)
		state_batch = [data[0] for data in minibatch]
		action_batch = [data[1] for data in minibatch]
		reward_batch = [data[2] for data in minibatch]
		nextState_batch = [data[3] for data in minibatch]

		# Step 2: calculate y 
		y_batch = []
		QValue_batch = self.QValueT.eval(feed_dict={self.stateInputT:nextState_batch})
		for i in range(0,BATCH_SIZE):
			terminal = minibatch[i][4]
			if terminal:
				y_batch.append(reward_batch[i])
			else:
				y_batch.append(reward_batch[i] + GAMMA * np.max(QValue_batch[i]))

		self.trainStep.run(feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, './savedweightsKeras/network' + '-dqn', global_step = self.timeStep)

		if self.timeStep % UPDATE_TIME == 0:
			self.copyTargetQNetwork()

		
	def setPerception(self,nextObservation,action,reward,terminal):
		newState = np.append(nextObservation,self.currentState[:,:,1:],axis = 2)
		self.replayMemory.append((self.currentState,action,reward,newState,terminal))
		if len(self.replayMemory) > REPLAY_MEMORY:
			self.replayMemory.popleft()
		if self.timeStep > OBSERVE:
			# Train the network
			self.trainQNetwork()

		# print info
		state = ""
		if self.timeStep <= OBSERVE:
			state = "observe"
		elif self.timeStep > OBSERVE and self.timeStep <= OBSERVE + EXPLORE:
			state = "explore"
		else:
			state = "train"
		if self.timeStep % 10000 == 0:
			print "TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon

		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		action = np.zeros(self.actions)
		action_index = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
				action[action_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1 # do nothing

		# change epsilon
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

		return action

	def setInitState(self,observation):
		self.currentState = np.stack((observation, observation, observation, observation), axis = 2)

	def weight_variable(self,shape):
		initial = tf.truncated_normal(shape, stddev = 0.01)
		return tf.Variable(initial)

	def bias_variable(self,shape):
		initial = tf.constant(0.01, shape = shape)
		return tf.Variable(initial)

	def conv2d(self,x, W, stride):
		return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID")

	def max_pool_2x2(self,x):
		return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
		
