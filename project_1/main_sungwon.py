import tensorflow as tf
import pandas as pd
import numpy as np
import pickle
import math
import matplotlib.pyplot as plt
from dataloader import DataLoader
from tensorflow.python.platform import gfile
from time import strftime, localtime, time


# properties
# General
# TODO : declare additional properties
# not fixed (change or add property as you like)
batch_size = 32
epoch_num = 2
print_every = 10

# fixed
metadata_path = 'dataset/track_metadata.csv'
# True if you want to train, False if you already trained your model
# TODO : IMPORTANT !!! Please change it to False when you submit your code
is_train_mode = True
# TODO : IMPORTANT !!! Please specify the path where your best model is saved
# example : checkpoint/run-0925-0348
checkpoint_path = 'checkpoint/run-1024-0117'
# 'track_genre_top' for project 1, 'listens' for project 2
label_column_name = 'track_genre_top'


n_input = 20 * 2498 #28*28 pixel
n_classs = 8 #0-9
image_height = 20
image_width = 2498
# convolutional layer property
conv_size = 3
n_filter1 = 32
n_filter2 = 64
n_filter3 = 128
# pooling layer property
pool_size = 2
# fully-connected layer property
fc_dim = 512
# Placeholder and variables
# TODO : declare placeholder and variables
X = tf.placeholder(tf.float32, [None, image_height, image_width])
y = tf.placeholder(tf.int64, [None, n_classs])
# Build model
# TODO : build your model here
regularizer = tf.contrib.layers.l2_regularizer(1e-4)
activation = tf.nn.relu
init = tf.contrib.layers.xavier_initializer()

w_fc = tf.get_variable("W_fc", shape=[3*313*n_filter3, fc_dim])#3*313 157 79 40
b_fc = tf.get_variable("b_fc", shape=[fc_dim])
w_out = tf.get_variable("W_out", shape=[fc_dim, n_classs])
b_out = tf.get_variable("b_out", shape=[n_classs])

x_reshaped = tf.reshape(X, [-1, image_height, image_width, 1])
c11 = tf.layers.conv2d(x_reshaped, n_filter1, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
c12 = tf.layers.conv2d(c11, n_filter1, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
mp1 = tf.layers.max_pooling2d(c12, pool_size, 2, "same")
c21 = tf.layers.conv2d(mp1, n_filter2, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
c22 = tf.layers.conv2d(c21, n_filter2, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
mp2 = tf.layers.max_pooling2d(c22, pool_size, 2, "same")
c31 = tf.layers.conv2d(mp2, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
c32 = tf.layers.conv2d(c31, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
mp3 = tf.layers.max_pooling2d(c32, pool_size, 2, "same")
# c41 = tf.layers.conv2d(mp3, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
# c42 = tf.layers.conv2d(c41, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
# mp4 = tf.layers.max_pooling2d(c32, pool_size, 2, "same")
# c51 = tf.layers.conv2d(mp4, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
# c52 = tf.layers.conv2d(c51, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
# mp5 = tf.layers.max_pooling2d(c32, pool_size, 2, "same")
# c61 = tf.layers.conv2d(mp5, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
# c62 = tf.layers.conv2d(c61, n_filter3, conv_size, 1, "same", activation = activation, kernel_initializer = init, kernel_regularizer = regularizer)
# mp6 = tf.layers.max_pooling2d(c62, pool_size, 2, "same")
if is_train_mode is not None:
    mp3 = tf.nn.dropout(mp3, 0.5)    
flat = tf.reshape(mp3, [-1, 3*313* n_filter3])
fc = tf.nn.relu(tf.matmul(flat, w_fc) + b_fc)
y_out = tf.matmul(fc, w_out) + b_out
# Loss and optimizer
# TODO : declare loss and optimizer operation
total_loss = tf.losses.softmax_cross_entropy(y,logits=y_out) 
mean_loss = tf.reduce_mean(total_loss)
optimizer = tf.train.AdamOptimizer(1e-4) 
extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(extra_update_ops):
	train_step = optimizer.minimize(mean_loss) 
#load data
with open("mfcc.pkl", 'rb') as mfcc:
	Xd = pickle.load(mfcc)
with open("y.pkl", 'rb') as ys:
	yd = pickle.load(ys)
# Train and evaluate
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	saver = tf.train.Saver()

	if is_train_mode:
		correct_prediction = tf.equal(tf.argmax(y_out,1), tf.argmax(y,1))
		print(tf.size(correct_prediction))
		accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))	
	    # shuffle indicies
		train_indicies = np.arange(Xd.shape[0])
		np.random.shuffle(train_indicies)
		iter_cnt = 0
		for e in range(epoch_num):
			# keep track of losses and accuracy
			correct = 0
			losses = []
			# make sure we iterate over the dataset once
			for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
				# generate indicies for the batch
				start_idx = (i*batch_size)%Xd.shape[0]
				idx = train_indicies[start_idx:start_idx+batch_size]
                # TODO:  do some train step code here
	            # create a feed dictionary for this batch
				feed_dict = {X: Xd[idx,:],
				             y: yd[idx] }
				# get batch size
				actual_batch_size = yd[idx].shape[0]
	            
				# have tensorflow compute loss and correct predictions
				# and (if given) perform a training step
				loss, corr, _ = sess.run([mean_loss,correct_prediction,train_step],feed_dict=feed_dict)

				# aggregate performance stats
				losses.append(loss*actual_batch_size)
				correct += np.sum(corr)

				# print every now and then
				if is_train_mode and (iter_cnt % print_every) == 0:
				    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
				          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
				iter_cnt += 1
			total_correct = correct/Xd.shape[0]
			total_loss = np.sum(losses)/Xd.shape[0]
			print("Epoch {2}, Overall loss = {0:.3g} and accuracy of {1:.3g}"\
              .format(total_loss,total_correct,e+1))
			plt.plot(losses)
			plt.grid(True)
			plt.title('Epoch {} Loss'.format(e+1))
			plt.xlabel('minibatch number')
			plt.ylabel('minibatch loss')
			plt.show()

		print('Training finished !')
		output_dir = checkpoint_path + '/run-%02d%02d-%02d%02d' % tuple(localtime(time()))[1:5]
		if not gfile.Exists(output_dir):
		    gfile.MakeDirs(output_dir)
		saver.save(sess, output_dir)
		print('Model saved in file : %s' % output_dir)
	else:
        # skip training and restore graph for validation test
		saver.restore(sess, checkpoint_path)

    # Validation
	# validation_dataloader = DataLoader(file_path='dataset/track_metadata.csv', batch_size=batch_size, label_column_name=label_column_name, is_training=False)
	#load data
	# if not is_train_mode:
		with open("mfcc_val.pkl", 'rb') as mfcc:
			Xd_val = pickle.load(mfcc)
		with open("y_val.pkl", 'rb') as ys:
			yd_val = pickle.load(ys)
	    # shuffle indicies
		train_indicies = np.arange(Xd_val.shape[0])
		np.random.shuffle(train_indicies)
		iter_cnt = 0			
		for e in range(epoch_num):
			# keep track of losses and accuracy
			correct = 0
			losses = []
			# make sure we iterate over the dataset once
			for i in range(int(math.ceil(Xd_val.shape[0]/batch_size))):
				# generate indicies for the batch
				start_idx = (i*batch_size)%Xd_val.shape[0]
				idx = train_indicies[start_idx:start_idx+batch_size]
                # TODO:  do some train step code here
	            # create a feed dictionary for this batch
				feed_dict = {X: Xd_val[idx,:],
				             y: yd_val[idx] }
				# get batch size
				actual_batch_size = yd_val[idx].shape[0]
	            
				# have tensorflow compute loss and correct predictions
				# and (if given) perform a training step
				loss = sess.run(mean_loss, feed_dict=feed_dict)

				# aggregate performance stats
				losses.append(loss*actual_batch_size)
				corr = 1
				# correct += np.sum(corr)
				print(loss)

				# print every now and then
				if is_train_mode and (iter_cnt % print_every) == 0:
				    print("Iteration {0}: with minibatch training loss = {1:.3g} and accuracy of {2:.2g}"\
				          .format(iter_cnt,loss,np.sum(corr)/actual_batch_size))
				iter_cnt += 1        # TODO : do some loss calculation here
        # average_cost += loss/validation_dataloader.num_batch
		print('Validation loss : %f' % losses)

    # accuracy test example
    # TODO :
    # pred = tf.nn.softmax(<your network output logit object>)
    # correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # accuracy_op = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    # avg_accuracy = 0
    # for i in range(validation_dataloader.num_batch):
    #batch_x, batch_y = validation_dataloader.next_batch()
    #acc = accuracy_op.eval({x:batch_x, y: batch_y})
    # avg_accuracy += acc / validation_dataloader.num_batch
    # print("Average accuracy on validation set ", avg_accuracy)
