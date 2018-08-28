# -----------------------------
# File: Deep Q-Learning Algorithm
# Author: Flood Sung
# Date: 2016.3.21
# -----------------------------

import tensorflow as tf 
import numpy as np 
import os
import random
from collections import deque, namedtuple

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

class BrainDQN:

	def __init__(self,actions):
		# init replay memory
		self.replayMemory = deque()
		# init some parameters
		self.timeStep = 0
		self.epsilon = INITIAL_EPSILON
		self.actions = actions
        
		self.avg_cost = []
		self.game_cost = []
        
		# init Q network
		self.stateInput, self.QValue, self.W_conv1, self.b_conv1, self.W_conv2, self.b_conv2, self.W_conv3, self.b_conv3, self.W_conv4, self.b_conv4, self.W_conv5, self.b_conv5, self.W_conv6, self.b_conv6, self.W_fc1, self.b_fc1, self.W_fc2, self.b_fc2, self.W_fc3, self.b_fc3, self.W_fc4, self.b_fc4 = self.createResNetQNetwork()
#		self.stateInput,self.QValue,self.W_conv1,self.b_conv1,self.W_conv2,self.b_conv2,self.W_conv3,self.b_conv3,self.W_fc1,self.b_fc1,self.W_fc2,self.b_fc2 = self.createQNetwork()

#		# init Target Q Network
#		self.stateInputT,self.QValueT,self.W_conv1T,self.b_conv1T,self.W_conv2T,self.b_conv2T,self.W_conv3T,self.b_conv3T,self.W_fc1T,self.b_fc1T,self.W_fc2T,self.b_fc2T = self.createQNetwork()
		self.stateInputT, self.QValueT, self.W_conv1T, self.b_conv1T, self.W_conv2T, self.b_conv2T, self.W_conv3T, self.b_conv3T, self.W_conv4T, self.b_conv4T, self.W_conv5T, self.b_conv5T, self.W_conv6T, self.b_conv6T, self.W_fc1T, self.b_fc1T, self.W_fc2T, self.b_fc2T, self.W_fc3T, self.b_fc3T, self.W_fc4T, self.b_fc4T = self.createResNetQNetwork("T")

#		self.copyTargetQNetworkOperation = [self.W_conv1T.assign(self.W_conv1),self.b_conv1T.assign(self.b_conv1),self.W_conv2T.assign(self.W_conv2),self.b_conv2T.assign(self.b_conv2),self.W_conv3T.assign(self.W_conv3),self.b_conv3T.assign(self.b_conv3),self.W_fc1T.assign(self.W_fc1),self.b_fc1T.assign(self.b_fc1),self.W_fc2T.assign(self.W_fc2),self.b_fc2T.assign(self.b_fc2)]

		self.copyTargetQNetworkOperation = [tf.assign(self.W_conv1T, self.W_conv1), tf.assign(self.b_conv1T, self.b_conv1), tf.assign(self.W_conv2T, self.W_conv2), tf.assign(self.b_conv2T, self.b_conv2T), tf.assign(self.W_conv3T, self.W_conv3), tf.assign(self.b_conv3T, self.b_conv3), tf.assign(self.W_conv4T, self.W_conv4), tf.assign(self.b_conv4T, self.b_conv4), tf.assign(self.W_conv5T, self.W_conv5), tf.assign(self.b_conv5T, self.b_conv5), tf.assign(self.W_conv6T, self.W_conv6), tf.assign(self.b_conv6T, self.b_conv6), tf.assign(self.W_fc1T, self.W_fc1), tf.assign(self.b_fc1T, self.b_fc1), tf.assign(self.W_fc2T, self.W_fc2), tf.assign(self.b_fc2T, self.b_fc2), tf.assign(self.W_fc3T, self.W_fc3), tf.assign(self.b_fc3T, self.b_fc3), tf.assign(self.W_fc4T, self.W_fc4), tf.assign(self.b_fc4T, self.b_fc4)]


		self.createTrainingMethod()

		# saving and loading networks
		self.saver = tf.train.Saver()
		self.session = tf.InteractiveSession()
		self.session.run(tf.global_variables_initializer())
		checkpoint = tf.train.get_checkpoint_state("./savedweights_resnet")
		if checkpoint and checkpoint.model_checkpoint_path:
				self.saver.restore(self.session, checkpoint.model_checkpoint_path)
				print "Successfully loaded:", checkpoint.model_checkpoint_path
		else:
				print "Could not find old network weights"


	def createResNetQNetwork(self, prefix=""):
		"""Builds a residual network.
		   prefix = name to prefix to the scope of all variables in this method.
		"""

		# Configurations for each bottleneck group.
		BottleneckGroup = namedtuple('BottleneckGroup',['num_blocks', 'num_filters', 'bottleneck_size'])
		groups = [BottleneckGroup(1, 32, 16)]

		stateInput = tf.placeholder("float",[None,65,160,4])
  
		# First convolution expands to 64 channels
		with tf.variable_scope(prefix + 'conv_layer1'):
			net = tf.layers.conv2d(
			stateInput,
			filters=16,
			kernel_size=6,
			strides=3,    
			activation=tf.nn.relu)
			W_conv1 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/kernel:0')
			b_conv1 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/bias:0')
			print("W_conv1:", W_conv1)
			print("b_conv1:", b_conv1)
            
		# Max pool
		net = tf.layers.max_pooling2d(net, pool_size=2, strides=2, padding='same')
		pool_out_shape = net.get_shape().as_list()
		print("Shape of the output of pool:", pool_out_shape[0], pool_out_shape[1], pool_out_shape[2], pool_out_shape[3])
		pool_total_params = pool_out_shape[1] * pool_out_shape[2] * pool_out_shape[3]
		print("Total params:", pool_total_params)
        
		# First chain of resnets
		with tf.variable_scope(prefix + 'conv_layer2'):
			net = tf.layers.conv2d(
			net,
			filters=groups[0].num_filters,
			kernel_size=1,
			padding='valid')
			W_conv2 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/kernel:0')
			b_conv2 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/bias:0')
			print("W_conv2:", W_conv2)
			print("b_conv2:", b_conv2)
            
		# Create the bottleneck groups, each of which contains `num_blocks`
		# bottleneck groups.
		for group_i, group in enumerate(groups):
			for block_i in range(group.num_blocks):
				name = 'group_%d/block_%d' % (group_i, block_i)

				# 1x1 convolution responsible for reducing dimension
				with tf.variable_scope(prefix + name + '/conv_in'):
					conv = tf.layers.conv2d(
					net,
					filters=group.num_filters,
					kernel_size=1,
					padding='valid',
					activation=tf.nn.relu)
					W_conv3 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/kernel:0')
					b_conv3 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/bias:0')
					print("W_conv3:", W_conv3)
					print("b_conv3:", b_conv3)


				with tf.variable_scope(prefix + name + '/conv_bottleneck'):
					conv = tf.layers.conv2d(
					conv,
					filters=group.bottleneck_size,
					kernel_size=3,
					padding='same',
					activation=tf.nn.relu)
					W_conv4 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/kernel:0')
					b_conv4 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/bias:0')
					print("W_conv4:", W_conv4)
					print("b_conv4:", b_conv4)

				# 1x1 convolution responsible for restoring dimension
				with tf.variable_scope(prefix + name + '/conv_out'):
					input_dim = net.get_shape()[-1].value
					conv = tf.layers.conv2d(
					conv,
					filters=input_dim,
					kernel_size=1,
					padding='valid',
					activation=tf.nn.relu)
					W_conv5 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/kernel:0')
					b_conv5 = tf.get_default_graph().get_tensor_by_name(os.path.split(conv.name)[0] + '/bias:0')
					print("W_conv5:", W_conv5)
					print("b_conv5:", b_conv5)

				# shortcut connections that turn the network into its counterpart
				# residual function (identity shortcut)
				net = conv + net

			try:
				# upscale to the next group size
				next_group = groups[group_i + 1]
				with tf.variable_scope(prefix + 'block_%d/conv_upscale' % group_i):
					net = tf.layers.conv2d(
						net,
						filters=next_group.num_filters,
						kernel_size=1,
						padding='same',
						activation=None,
						bias_initializer=None)
			except IndexError:
				pass

		# Last convolution expands before FC layers maps to total size of 1056
		with tf.variable_scope(prefix + 'conv_layer6'):
			net = tf.layers.conv2d(
			net,
			filters=16,
			kernel_size=1,
			activation=tf.nn.relu)
			W_conv6 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/kernel:0')
			b_conv6 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/bias:0')
			print("W_conv6:", W_conv6)
			print("b_conv6:", b_conv6)
			conv6_shape = net.get_shape().as_list()
			print("Shape of the output of conv6:", conv6_shape[0], conv6_shape[1], conv6_shape[2], conv6_shape[3])
			conv6_total_params = conv6_shape[1] * conv6_shape[2] * conv6_shape[3]
			print("Total params:", conv6_total_params)

		# Reshape to pass to fully connected layers.            
		net_flat = tf.reshape(net,[-1,conv6_total_params])
            
		# Bottleneck to total size of 1056
		with tf.variable_scope(prefix + 'fc_layer1'):
			net = tf.layers.dense(net_flat, 1056, activation=tf.nn.relu)
			W_fc1 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/kernel:0')
			b_fc1 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/bias:0')
			print("W_fc1:", W_fc1)
			print("b_fc1:", b_fc1)
			fc1_shape = net.get_shape().as_list()
			print("fc1-shape:", fc1_shape)

		# Bottleneck to total size of 176
		with tf.variable_scope(prefix + 'fc_layer2'):
			net = tf.layers.dense(net, 176, activation=tf.nn.relu)
			W_fc2 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/kernel:0')
			b_fc2 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/bias:0')
			print("W_fc2:", W_fc2)
			print("b_fc2:", b_fc2)
			fc2_shape = net.get_shape().as_list()
			print("fc2-shape:", fc2_shape)

		# Bottleneck to total size of 44
		with tf.variable_scope(prefix + 'fc_layer3'):
			net = tf.layers.dense(net, 44, activation=tf.nn.relu)
			W_fc3 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/kernel:0')
			b_fc3 = tf.get_default_graph().get_tensor_by_name(os.path.split(net.name)[0] + '/bias:0')
			print("W_fc3:", W_fc3)
			print("b_fc3:", b_fc3)
			fc3_shape = net.get_shape().as_list()
			print("fc3-shape:", fc3_shape)

		# Bottleneck to total size of 4
		with tf.variable_scope(prefix + 'fc_layer4'):
			QValue = tf.layers.dense(net, self.actions, activation=None)
			W_fc4 = tf.get_default_graph().get_tensor_by_name(os.path.split(QValue.name)[0] + '/kernel:0')
			b_fc4 = tf.get_default_graph().get_tensor_by_name(os.path.split(QValue.name)[0] + '/bias:0')
			print("W_fc4:", W_fc4)
			print("b_fc4:", b_fc4)
			qvalue_shape = QValue.get_shape().as_list()
			print("qvalue-shape:", qvalue_shape)
        
		return stateInput, QValue, W_conv1, b_conv1, W_conv2, b_conv2, W_conv3, b_conv3, W_conv4, b_conv4, W_conv5, b_conv5, W_conv6, b_conv6, W_fc1, b_fc1, W_fc2, b_fc2, W_fc3, b_fc3, W_fc4, b_fc4
                
                
#	def createQNetwork(self):
#		# network weights
#		W_conv1 = self.weight_variable([8,8,4,32])
#		b_conv1 = self.bias_variable([32])
#
#		W_conv2 = self.weight_variable([4,4,32,64])
#		b_conv2 = self.bias_variable([64])
#
#		W_conv3 = self.weight_variable([3,3,64,64])
#		b_conv3 = self.bias_variable([64])
#
#		W_fc1 = self.weight_variable([3136,512])
#		b_fc1 = self.bias_variable([512])
#
#		W_fc2 = self.weight_variable([512,self.actions])
#		b_fc2 = self.bias_variable([self.actions])
#
#		# input layer
#
#		stateInput = tf.placeholder("float",[None,84,84,4])
#
#		# hidden layers
#		h_conv1 = tf.nn.relu(self.conv2d(stateInput,W_conv1,4) + b_conv1)
#		#h_pool1 = self.max_pool_2x2(h_conv1)
#
#		h_conv2 = tf.nn.relu(self.conv2d(h_conv1,W_conv2,2) + b_conv2)
#
#		h_conv3 = tf.nn.relu(self.conv2d(h_conv2,W_conv3,1) + b_conv3)
#		h_conv3_shape = h_conv3.get_shape().as_list()
#		print "dimension:",h_conv3_shape[1]*h_conv3_shape[2]*h_conv3_shape[3]
#		h_conv3_flat = tf.reshape(h_conv3,[-1,3136])
#		h_fc1 = tf.nn.relu(tf.matmul(h_conv3_flat,W_fc1) + b_fc1)
#
#		# Q Value layer
#		QValue = tf.matmul(h_fc1,W_fc2) + b_fc2
#		qvalue_shape = QValue.get_shape().as_list()
#		print("qvalue-shape:", qvalue_shape)
#
#		return stateInput,QValue,W_conv1,b_conv1,W_conv2,b_conv2,W_conv3,b_conv3,W_fc1,b_fc1,W_fc2,b_fc2

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

		train, cost = self.session.run([self.trainStep, self.cost], feed_dict={
			self.yInput : y_batch,
			self.actionInput : action_batch,
			self.stateInput : state_batch
			})                
              
		self.game_cost.append(cost)                                       
		if terminal:                                       
			self.avg_cost.append(np.mean(self.game_cost))
#			print(np.mean)
			self.game_cost = []                                       
                                       
#		self.trainStep.run(feed_dict={
#			self.yInput : y_batch,
#			self.actionInput : action_batch,
#			self.stateInput : state_batch
#			})

		# save network every 100000 iteration
		if self.timeStep % 10000 == 0:
			self.saver.save(self.session, './savedweights_resnet/network' + '-dqn', global_step = self.timeStep)

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
			action[0] = 1 # do nothingf

		# change episilon
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
		
