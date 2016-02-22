#!/usr/bin/env python

"""
Usage example employing Lasagne for digit recognition using the MNIST dataset.

This example is deliberately structured as a long flat file, focusing on how
to use Lasagne, instead of focusing on writing maximally modular and reusable
code. It is used as the foundation for the introductory Lasagne tutorial:
http://lasagne.readthedocs.org/en/latest/user/tutorial.html

More in-depth examples and reproductions of paper results are maintained in
a separate repository: https://github.com/Lasagne/Recipes
"""

from __future__ import print_function

import sys
import os
import time

import numpy as np
import theano
import theano.tensor as T

import lasagne


# ################## Download and prepare the MNIST dataset ##################
# This is just some way of getting the MNIST dataset from an online location
# and loading it into numpy arrays. It doesn't involve Lasagne at all.

def load_dataset():
	# We first define a download function, supporting both Python 2 and 3.
	if sys.version_info[0] == 2:
		from urllib import urlretrieve
	else:
		from urllib.request import urlretrieve

	def download(filename, source='http://yann.lecun.com/exdb/mnist/'):
		print("Downloading %s" % filename)
		urlretrieve(source + filename, filename)

	# We then define functions for loading MNIST images and labels.
	# For convenience, they also download the requested files if needed.
	import gzip

	def load_mnist_images(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the inputs in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=16)
		# The inputs are vectors now, we reshape them to monochrome 2D images,
		# following the shape convention: (examples, channels, rows, columns)
		data = data.reshape(-1, 1, 28, 28)
		# The inputs come as bytes, we convert them to float32 in range [0,1].
		# (Actually to range [0, 255/256], for compatibility to the version
		# provided at http://deeplearning.net/data/mnist/mnist.pkl.gz.)
		print (data[:10][0] / np.float32(256))
		#
		print ((data/np.float32(256)).dtype)
		
		return data / np.float32(256)

	def load_mnist_labels(filename):
		if not os.path.exists(filename):
			download(filename)
		# Read the labels in Yann LeCun's binary format.
		with gzip.open(filename, 'rb') as f:
			data = np.frombuffer(f.read(), np.uint8, offset=8)
		# The labels are vectors of integers now, that's exactly what we want.
		
		print (data[:10])
		
		return data

		
	def load_15_data(filename):
		if not os.path.exists(filename):
			print ("Input data not exisits error[%s]" % filename)
			sys.exit()
		data = []
		with open(filename, 'rb') as f:
			#data = np.frombuffer(f.read(), np.uint8, offset=0)
			lines = f.readlines()
		#size = 15
		global size
		for first in lines:
			size = len(first.split(','))
			break
		for line in lines:
			data.append(line.split(','))
			
			
		print ('total element of one line : %s' % size)
		data = np.array(data)
		data = data.astype(np.float32)
		
		#data = np.ndarray(data,dtype=float,order='F')
		#data = np.array(data,dtype=np.float32)
		data = data * np.float32(100)
		#data = data.astype(np.int32)
		#data = np.array(data,dtype=np.int32)
		#print (data[:10])
		#data = np.int32(data)
		print ("15 : %s" % len(data))
		#print (data[0])
		data = data.reshape(-1, 1,size)
		#data = np.absolute(data)
		#data = data / data.max(axis=0)
		print (data[:2])
		#print (np.int32(data[:10] * np.int32(100)))
		#data,np.int32(data * np.int32(100))
		#return np.int32(data * np.int32(100))
		print ("15 dtype: %s" % data.dtype)
		return data
		
	def load_15_lable_data(filename):
		if not os.path.exists(filename):
			print ("Input data not exisits error")
			sys.exit()
		# Read the labels in Yann LeCun's binary format.
		with open(filename, 'rb') as f:
			#data = np.frombuffer(f.read(), np.uint8, offset=0)
			data = f.readlines()
			
		#data = np.ndarray(data,dtype=float,order='F')
		data = np.array(data,dtype=np.float32)
		#import random
		#print (data[:10])
		#random.shuffle(data)
		#print (data[:10])
		print ("Label : %s" % len(data))
		#print (type(data))
		decimal = 100
		print ( np.int32(data[:10] * np.int32(decimal)))
		#print ('hey')
		# The labels are vectors of integers now, that's exactly what we want.
		return np.int32(data * np.int32(decimal))
		
		
	# We can now download and read the training and test set images and labels.
	#X_train = load_15_data('tonyu_data/standard/train_f.15')
	X_train_IN = load_15_data('yiming_data/IN')
	X_train_OUT = load_15_data('yiming_data/OUT')
	#y_train = load_15_lable_data('yiming_data/train_f.label')
	num_l = 100000
	y_train_IN = np.empty(num_l)
	y_train_IN.fill(1)
	y_train_IN = np.array(y_train_IN,dtype=np.int32)
	y_train_OUT = np.empty(num_l)
	y_train_OUT.fill(0)
	y_train_OUT = np.array(y_train_OUT,dtype=np.int32)
	
	from sklearn.utils import shuffle
	##combine in and out together
	X_train_tmp = np.concatenate((X_train_IN,X_train_OUT),axis=0)
	#y_train_tmp = y_train_IN + y_train_OUT
	y_train_tmp = np.concatenate((y_train_IN,y_train_OUT ),axis=0)
	
	print ("OMG")
	print (len(X_train_tmp))
	print (X_train_tmp[:3])
	X_train,y_train = shuffle(X_train_tmp, y_train_tmp, random_state=0)
	print (type(X_train))
	#print (X_train[:50])
	print (y_train[:50])
	##shuffle two arrays
	
	#print (y_train[:100])
	#print (len(X_train))
	#print (len(y_train))
	#print (y_train_IN[:50])
	#X_test = load_15_data('tonyu_data/standard/test_f.15')
	global test_file_name
	X_test = load_15_data('yiming_data/'+test_file_name)
	#y_test = load_15_lable_data('yiming_data/test_f.label')
	#print("WHAT")
	y_test = np.empty(num_l)
	y_test[:num_l/2].fill(0)
	y_test[num_l/2:].fill(1)
	y_test = np.array(y_test,dtype=np.int32)
	
	#arr = np.arange(100).reshape(1)
	'''
	print ("Used random_label for test_set")
	arr = np.linspace(0,100, num=len(X_test), retstep=True,dtype='int32')[0]
	np.random.shuffle(arr)
	y_test = arr'''
	#print ('len %s ' % len(arr))
	#print (arr[:40])
	#import sys
	#sys.exit()
	
	
	#X_train = load_mnist_images('train-images-idx3-ubyte.gz')
	#y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
	#X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
	#y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')

	# We reserve the last 10000 training examples for validation.
	test_per = 0.2
	num = int(len(X_train) - len(X_train)*test_per)
	print ("Training data : %s" % num)
	X_train, X_val = X_train[:-num], X_train[-num:]
	y_train, y_val = y_train[:-num], y_train[-num:]

	# We just return all the arrays in order, as expected in main().
	# (It doesn't matter how we do this as long as we can read them again.)
	return X_train, y_train, X_val, y_val, X_test, y_test

global test_file_name
test_f = 'Test'
#test_f = '100best_dev'
#test_f = '2best_test'
test_file_name = test_f
#test_file_name = test_f+'.vector'
s_b_file = test_f+'.sb'

if os.path.isfile(s_b_file):
	os.remove(s_b_file)

# ##################### Build the neural network model #######################
# This script supports three types of models. For each one, we define a
# function that takes a Theano variable representing the input and returns
# the output layer of a neural network model built in Lasagne.

def build_mlp(input_var=None):
	# This creates an MLP of two hidden layers of 800 units each, followed by
	# a softmax output layer of 10 units. It applies 20% dropout to the input
	# data and 50% dropout to the hidden layers.
	global size
	# Input layer, specifying the expected input shape of the network
	# (unspecified batchsize, 1 channel, 28 rows and 28 columns) and
	# linking it to the given Theano variable `input_var`, if any:
	#l_in = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
	l_in = lasagne.layers.InputLayer(shape=(None, 1,size),
									 input_var=input_var)

	# Apply 20% dropout to the input data:
	l_in_drop = lasagne.layers.DropoutLayer(l_in, p=0.2)

	# Add a fully-connected layer of 800 units, using the linear rectifier, and
	# initializing weights with Glorot's scheme (which is the default anyway):
	l_hid1 = lasagne.layers.DenseLayer(
			l_in_drop, num_units=200,
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())

	# We'll now add dropout of 50%:
	l_hid1_drop = lasagne.layers.DropoutLayer(l_hid1, p=0.5)

	# Another 800-unit layer:
	l_hid2 = lasagne.layers.DenseLayer(
			l_hid1_drop, num_units=200,
			nonlinearity=lasagne.nonlinearities.rectify)

	# 50% dropout again:
	l_hid2_drop = lasagne.layers.DropoutLayer(l_hid2, p=0.5)

	# Finally, we'll add the fully-connected output layer, of 10 softmax units:
	l_out = lasagne.layers.DenseLayer(
			l_hid2_drop, num_units=2,
			nonlinearity=lasagne.nonlinearities.softmax)

	# Each layer is linked to its incoming layer(s), so we only need to pass
	# the output layer to give access to a network in Lasagne:
	return l_out


def build_custom_mlp(input_var=None, depth=2, width=800, drop_input=.2,
					 drop_hidden=.5):
	# By default, this creates the same network as `build_mlp`, but it can be
	# customized with respect to the number and size of hidden layers. This
	# mostly showcases how creating a network in Python code can be a lot more
	# flexible than a configuration file. Note that to make the code easier,
	# all the layers are just called `network` -- there is no need to give them
	# different names if all we return is the last one we created anyway; we
	# just used different names above for clarity.

	# Input layer and dropout (with shortcut `dropout` for `DropoutLayer`):
	network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
										input_var=input_var)
	if drop_input:
		network = lasagne.layers.dropout(network, p=drop_input)
	# Hidden layers and dropout:
	nonlin = lasagne.nonlinearities.rectify
	for _ in range(depth):
		network = lasagne.layers.DenseLayer(
				network, width, nonlinearity=nonlin)
		if drop_hidden:
			network = lasagne.layers.dropout(network, p=drop_hidden)
	# Output layer:
	softmax = lasagne.nonlinearities.softmax
	network = lasagne.layers.DenseLayer(network, 10, nonlinearity=softmax)
	return network

def build_cnn(input_var=None):
	# As a third model, we'll create a CNN of two convolution + pooling stages
	# and a fully-connected hidden layer in front of the output layer.

	# Input layer, as usual:
	network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
										input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	# Max-pooling layer of factor 2 in both dimensions:
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	network = lasagne.layers.Conv2DLayer(
			network, num_filters=32, filter_size=(5, 5),
			nonlinearity=lasagne.nonlinearities.rectify)
	network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units=256,
			nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax)

	return network
	
def build_custom_cnn(input_var=None):
	# As a third model, we'll create a CNN of two convolution + pooling stages
	# and a fully-connected hidden layer in front of the output layer.

	# Input layer, as usual:
	#network = lasagne.layers.InputLayer(shape=(None, 1, 28, 28),
										#input_var=input_var)
	network = lasagne.layers.InputLayer(shape=(None, 1,10),
										input_var=input_var)
	# This time we do not apply input dropout, as it tends to work less well
	# for convolutional layers.

	# Convolutional layer with 32 kernels of size 5x5. Strided and padded
	# convolutions are supported as well; see the docstring.
	network = lasagne.layers.Conv1DLayer(
			#network, num_filters=32, filter_size=(5, 5),
			network, num_filters=32, filter_size=5,
			nonlinearity=lasagne.nonlinearities.rectify,
			W=lasagne.init.GlorotUniform())
	# Expert note: Lasagne provides alternative convolutional layers that
	# override Theano's choice of which implementation to use; for details
	# please see http://lasagne.readthedocs.org/en/latest/user/tutorial.html.

	# Max-pooling layer of factor 2 in both dimensions:
	#network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

	# Another convolution with 32 5x5 kernels, and another 2x2 pooling:
	network = lasagne.layers.Conv1DLayer(
			#network, num_filters=32, filter_size=(5, 5),
			network, num_filters=32, filter_size=5,
			nonlinearity=lasagne.nonlinearities.rectify)
	#network = lasagne.layers.MaxPool2DLayer(network, pool_size=(2, 2))
	network = lasagne.layers.MaxPool1DLayer(network, pool_size=2)

	# A fully-connected layer of 256 units with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units=100,
			nonlinearity=lasagne.nonlinearities.rectify)

	# And, finally, the 10-unit output layer with 50% dropout on its inputs:
	network = lasagne.layers.DenseLayer(
			lasagne.layers.dropout(network, p=.5),
			num_units=10,
			nonlinearity=lasagne.nonlinearities.softmax)

	return network


# ############################# Batch iterator ###############################
# This is just a simple helper function iterating over training data in
# mini-batches of a particular size, optionally in random order. It assumes
# data is available as numpy arrays. For big datasets, you could load numpy
# arrays as memory-mapped files (np.load(..., mmap_mode='r')), or write your
# own custom data iteration function. For small datasets, you can also copy
# them to GPU at once for slightly improved performance. This would involve
# several changes in the main program, though, and is not demonstrated here.

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
	#assert len(inputs) == len(targets)
	if shuffle:
		indices = np.arange(len(inputs))
		np.random.shuffle(indices)
	for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
		if shuffle:
			excerpt = indices[start_idx:start_idx + batchsize]
		else:
			excerpt = slice(start_idx, start_idx + batchsize)
		yield inputs[excerpt], targets[excerpt]


# ############################## Main program ################################
# Everything else will be handled in our main program now. We could pull out
# more functions to better separate the code, but it wouldn't make it any
# easier to read.

def main(model='mlp', num_epochs=500):
	# Load the dataset
	print("Loading data...")
	X_train, y_train, X_val, y_val, X_test, y_test = load_dataset()

	# Prepare Theano variables for inputs and targets
	#input_var = T.tensor4('inputs')
	input_var = T.tensor3('inputs')
	input_var = T.tensor3('inputs',dtype='float32')
	#input_var = T.tensor3('inputs',dtype='int32')
	#input_var = T.matrix('inputs')
	
	print ("input : %s' " % input_var.dtype)
	
	target_var = T.ivector('targets')

	# Create neural network model (depending on first command line parameter)
	print("Building model and compiling functions...")
	if model == 'mlp':
		network = build_mlp(input_var)
	elif model.startswith('custom_mlp:'):
		depth, width, drop_in, drop_hid = model.split(':', 1)[1].split(',')
		network = build_custom_mlp(input_var, int(depth), int(width),
								   float(drop_in), float(drop_hid))
	elif model == 'cnn':
		network = build_cnn(input_var)
	elif model == 'c_cnn':
		network = build_custom_cnn(input_var)
	else:
		print("Unrecognized model type %r." % model)
		return

	# Create a loss expression for training, i.e., a scalar objective we want
	# to minimize (for our multi-class problem, it is the cross-entropy loss):
	prediction = lasagne.layers.get_output(network)
	loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
	loss = loss.mean()
	# We could add some weight decay as well here, see lasagne.regularization.

	# Create update expressions for training, i.e., how to modify the
	# parameters at each training step. Here, we'll use Stochastic Gradient
	# Descent (SGD) with Nesterov momentum, but Lasagne offers plenty more.
	params = lasagne.layers.get_all_params(network, trainable=True)
	updates = lasagne.updates.nesterov_momentum(
			loss, params, learning_rate=0.01, momentum=0.9)

	# Create a loss expression for validation/testing. The crucial difference
	# here is that we do a deterministic forward pass through the network,
	# disabling dropout layers.
	test_prediction = lasagne.layers.get_output(network, deterministic=True)
	
	#test_pre = test_prediction
	test_pre = T.argmax(test_prediction, axis=1)
	
	
	test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
															target_var)
	test_loss = test_loss.mean()
	# As a bonus, also create an expression for the classification accuracy:
	test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
					  dtype=theano.config.floatX)

	# Compile a function performing a training step on a mini-batch (by giving
	# the updates dictionary) and returning the corresponding training loss:
	train_fn = theano.function([input_var, target_var], loss, updates=updates)

	# Compile a second function computing the validation loss and accuracy:
	val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
	
	pre_fn = theano.function([input_var],test_pre)

	# Finally, launch the training loop.
	#import sys
	#sys.exit()
	
	print("Starting training...")
	# We iterate over epochs:
	for epoch in range(num_epochs):
		# In each epoch, we do a full pass over the training data:
		train_err = 0
		train_batches = 0
		start_time = time.time()
		for batch in iterate_minibatches(X_train, y_train, 500, shuffle=True):
			inputs, targets = batch
			train_err += train_fn(inputs, targets)
			train_batches += 1

		# And a full pass over the validation data:
		val_err = 0
		val_acc = 0
		val_batches = 0
		for batch in iterate_minibatches(X_val, y_val, 500, shuffle=False):
			inputs, targets = batch
			err, acc = val_fn(inputs, targets)
			val_err += err
			val_acc += acc
			val_batches += 1

		# Then we print the results for this epoch:
		print("Epoch {} of {} took {:.3f}s".format(
			epoch + 1, num_epochs, time.time() - start_time))
		print("  training loss:\t\t{:.6f}".format(train_err / train_batches))
		print("  validation loss:\t\t{:.6f}".format(val_err / val_batches))
		print("  validation accuracy:\t\t{:.2f} %".format(
			val_acc / val_batches * 100))

	# After training, we compute and print the test error:
	test_err = 0
	test_acc = 0
	test_batches = 0
	for batch in iterate_minibatches(X_test, y_test, 500, shuffle=False):
		inputs, targets = batch
		err, acc = val_fn(inputs, targets)
		test_err += err
		test_acc += acc
		test_batches += 1
		
		'''print ("input_test")
		print (inputs)
		print ("target_test")
		print (len(targets))
		print (targets)
		print ("display prediction scores")
		pre = pre_fn(inputs)
		print (len(pre))
		#print (pre[0])
		print (pre)'''
		
		global s_b_file
		'''
		print ("starting writing sentence_blue into file")
		for sentence_blue in pre:
			with open(s_b_file,'a') as f:
				f.write('%s\n' % sentence_blue)
		'''
		
	print("Final results:")
	print("  test loss:\t\t\t{:.6f}".format(test_err / test_batches))
	print("  test accuracy:\t\t{:.2f} %".format(
		test_acc / test_batches * 100))

	# Optionally, you could now dump the network weights to a file like this:
	# np.savez('model.npz', *lasagne.layers.get_all_param_values(network))
	#
	# And load them again later on like this:
	# with np.load('model.npz') as f:
	#	 param_values = [f['arr_%d' % i] for i in range(len(f.files))]
	# lasagne.layers.set_all_param_values(network, param_values)


if __name__ == '__main__':
	if ('--help' in sys.argv) or ('-h' in sys.argv):
		print("Trains a neural network on MNIST using Lasagne.")
		print("Usage: %s [MODEL [EPOCHS]]" % sys.argv[0])
		print()
		print("MODEL: 'mlp' for a simple Multi-Layer Perceptron (MLP),")
		print("	   'custom_mlp:DEPTH,WIDTH,DROP_IN,DROP_HID' for an MLP")
		print("	   with DEPTH hidden layers of WIDTH units, DROP_IN")
		print("	   input dropout and DROP_HID hidden dropout,")
		print("	   'cnn' for a simple Convolutional Neural Network (CNN).")
		print("EPOCHS: number of training epochs to perform (default: 500)")
	else:
		kwargs = {}
		if len(sys.argv) > 1:
			kwargs['model'] = sys.argv[1]
		if len(sys.argv) > 2:
			kwargs['num_epochs'] = int(sys.argv[2])
		main(**kwargs)
