# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf 
import numpy as np 
import random
from collections import deque
import os

import keras
from keras import models
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, Input
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
eps = 0.001


class BrainDQN3L:

	def __init__(self,actions):

		# Init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
        
		# Init cost tracking variables
		self.game_cost = []
		self.avg_cost = []
        
		# Init Q network
		self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3 = self.createQNetwork()

		# Init Target Q Network
		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T,self.W_fc3T,self.b_fc3T = self.createQNetwork("T")

		# Init copyTargetQNetworkOperation
		self.copyTargetQNetworkOperation = [tf.assign(self.W_conv1T,self.W_conv1), tf.assign(self.b_conv1T,self.b_conv1), tf.assign(self.W_conv2T,self.W_conv2), tf.assign(self.b_conv2T,self.b_conv2), tf.assign(self.W_conv3T,self.W_conv3), tf.assign(self.b_conv3T,self.b_conv3), tf.assign(self.W_fc1T,self.W_fc1), tf.assign(self.b_fc1T,self.b_fc1), tf.assign(self.W_fc2T,self.W_fc2), tf.assign(self.b_fc2T,self.b_fc2), tf.assign(self.W_fc3T,self.W_fc3), tf.assign(self.b_fc3T,self.b_fc3)]

		# Init trainingMethod
		self.createTrainingMethod()

		# Saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state("./savedweightsKeras_epsloss")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print("Successfully loaded:", checkpoint.model_checkpoint_path)
		else:
				print("Could not find old network weights")


	def createQNetwork(self, prefix=""):

		# Image dimensions (input) :
		width = 65
		height = 160
		depth = 4

		# input layer
		stateInput = Input(shape=(width, height, depth))
        
		h_conv1 = Conv2D(16, (6, 6), activation='relu', strides=(3,3), padding='valid')(stateInput)
		mpool = MaxPooling2D(pool_size=(2, 2), padding='valid')(h_conv1)
		h_conv2 = Conv2D(32, (4, 4), activation='relu', strides=(2,2), padding='valid')(mpool)
		h_conv3 = Conv2D(32, (2, 2), activation='relu', strides=(1,1), padding='valid')(h_conv2)
		h_conv3_flat = Flatten()(h_conv3)
		fc1 = Dense(176, activation='relu', kernel_regularizer= 'l2')(h_conv3_flat)
		fc2 = Dense(44, activation='relu', kernel_regularizer= 'l2')(fc1)
		QValue = Dense(6, activation='relu', kernel_regularizer= 'l2')(fc2)
        
		model = Model(inputs=stateInput,output=QValue)

		#print(len(model.layers))
		print(len(model.layers[1].get_weights()))
		print(model.layers[1].input_shape)
		#print(model.layers[1].output_shape)
		print(model.layers[2].input_shape)
		#print(model.layers[2].output_shape)
		print(model.layers[3].input_shape)
		#print(model.layers[3].output_shape)
		print(model.layers[4].input_shape)
		#print(model.layers[4].output_shape)
		print(model.layers[5].input_shape)
		#print(model.layers[5].output_shape)
		print(model.layers[6].input_shape)
		#print(model.layers[6].output_shape)
		print(model.layers[7].input_shape)
		#print(model.layers[7].output_shape)
		print(model.layers[8].input_shape)
		print(model.layers[8].output_shape)
        
		#print(len(model.layers[0].get_output_shape_at(1)))
		#print(len(model.layers[1].get_weights()[1]))
		with tf.variable_scope(prefix):
			#W_conv1 = tf.convert_to_tensor(model.layers[1].get_weights()[0], name='wacko', dtype=np.float32)
			#b_conv1 = tf.convert_to_tensor(model.layers[1].get_weights()[1])
			W_conv1 = tf.get_default_graph().get_tensor_by_name(os.path.split(h_conv1.name)[0] + '/kernel:0')
			b_conv1 = tf.get_default_graph().get_tensor_by_name(os.path.split(h_conv1.name)[0] + '/bias:0')
			print("W_conv1-shape:", W_conv1.get_shape().as_list())
			print("W_conv1:", W_conv1)

			W_conv2 = tf.get_default_graph().get_tensor_by_name(os.path.split(h_conv2.name)[0] + '/kernel:0')
			b_conv2 = tf.get_default_graph().get_tensor_by_name(os.path.split(h_conv2.name)[0] + '/bias:0')

			W_conv3 = tf.get_default_graph().get_tensor_by_name(os.path.split(h_conv3.name)[0] + '/kernel:0')
			b_conv3 = tf.get_default_graph().get_tensor_by_name(os.path.split(h_conv3.name)[0] + '/bias:0')

			W_fc1 = tf.get_default_graph().get_tensor_by_name(os.path.split(fc1.name)[0] + '/kernel:0')
			b_fc1 = tf.get_default_graph().get_tensor_by_name(os.path.split(fc1.name)[0] + '/bias:0')

			W_fc2 = tf.get_default_graph().get_tensor_by_name(os.path.split(fc2.name)[0] + '/kernel:0')
			b_fc2 = tf.get_default_graph().get_tensor_by_name(os.path.split(fc2.name)[0] + '/bias:0')

			W_fc3 = tf.get_default_graph().get_tensor_by_name(os.path.split(QValue.name)[0] + '/kernel:0')
			b_fc3 = tf.get_default_graph().get_tensor_by_name(os.path.split(QValue.name)[0] + '/bias:0')

		return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3


	def copyTargetQNetwork(self):
		self.session.run(self.copyTargetQNetworkOperation)


	def createTrainingMethod(self):
		self.actionInput = tf.placeholder("float",[None,self.actions])
		self.yInput = tf.placeholder("float", [None]) 
		Q_Action = tf.reduce_sum(tf.multiply(self.QValue, self.actionInput), reduction_indices = 1)
        
        # Possible new cost function ? Use epsilon tube loss as well ?
        #loss =  tf.reduce_sum(tf.square(self.yInput - Q_Action))
        #self.cost = tf.reduce_mean(loss)
        
		# Previous cost function
		#self.cost = tf.reduce_mean(tf.square(self.yInput - Q_Action))
		# New cost function using epsilon loss model
		self.cost = tf.maximum(0., tf.abs(tf.reduce_mean(tf.square(self.yInput - Q_Action))) - eps)
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

		# We use session.run in order to calculate the cost to keep track of performance
		train, cost = self.session.run([self.trainStep,self.cost], feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})
		# We append the cost to the game_cost list
		self.game_cost.append(cost)
		if terminal:
			# When the game is over, we take the average cost per game
			self.avg_cost.append(np.mean(self.game_cost))
#			print(np.mean(costs))
			self.game_cost = []
        
		# Training step, using the replay memory
#		self.trainStep.run(feed_dict={
#			self.yInput : y_batch,
#			self.actionInput : action_batch,
#			self.stateInput : state_batch
#			})

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, './savedweightsKeras_epsloss/network' + '-dqn', global_step = self.timeStep)

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
			print("TIMESTEP", self.timeStep, "/ STATE", state, "/ EPSILON", self.epsilon)

		self.currentState = newState
		self.timeStep += 1

	def getAction(self):
		QValue = self.QValue.eval(feed_dict= {self.stateInput:[self.currentState]})[0]
		action = np.zeros(self.actions)
		action_index = 0
		if self.timeStep % FRAME_PER_ACTION == 0:
			if random.random() <= self.epsilon:
				action_index = random.randrange(self.actions)
                
				# We have a one-hot encoded action
				action[action_index] = 1
			else:
				action_index = np.argmax(QValue)
				action[action_index] = 1
		else:
			action[0] = 1 # do nothing

		# Change epsilon, this defines the proportion of randomness of the agent's actions
		if self.epsilon > FINAL_EPSILON and self.timeStep > OBSERVE:
			self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON)/EXPLORE

        # We return an array that's one-hot encoded for the action
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
		
