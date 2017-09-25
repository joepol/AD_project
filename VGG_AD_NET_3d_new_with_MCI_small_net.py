import os
import numpy as np
import struct
import tensorflow as tf
import pandas as pd
import sys
import math
from PIL import Image

FLAGS = None
def weight_variable(shape, weight_name):
    # generates random values for initial weights
    #initial = tf.truncated_normal(shape, stddev=0.1)
    initializer = tf.get_variable(weight_name, shape,
                    initializer=tf.contrib.layers.xavier_initializer())
    #return tf.Variable(initializer)
    return initializer;

def bias_variable(shape, bias_name):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

def conv3d(x, W, layername):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x, layer_name):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1], strides=[1, 2, 2, 2, 1], padding='SAME')

def next_batch(num, data, labels):	## fix next batch for 3D
# Return a total of `num` random samples and labels. 
    idx = np.arange(0 , len(data))
    np.random.shuffle(idx)
    idx = idx[:num]
    data_shuffle = [data[ i] for i in idx]
    labels_shuffle = [labels[ i] for i in idx]
    return np.asarray(data_shuffle), np.asarray(labels_shuffle)

def next_batch_valid(num, data, labels, batch_num):	## fix next batch for 3D
# Return a total of `num` random samples and labels. 
    idx = np.arange(0 , len(data))
    #np.random.shuffle(idx)
    #idx = idx[:num]
    idx = idx[batch_num*j:num*(batch_num+1)]
    data_not_shuffled = [data[ i] for i in idx]
    labels_not_shuffled = [labels[ i] for i in idx]
    return np.asarray(data_not_shuffled), np.asarray(labels_not_shuffled)


# #######################################
            #Loading dataset#
# #######################################
#

#DIR = os.path.dirname(os.path.abspath(__file__))
DIR = os.getcwd() + "/"
print("DIR folder is ", DIR)
train_tmp = np.load('../img_array_train_6k_1.npy')#[:5952]
# use the commands train_all.size or train_all.shape to get info
for i in range(2,18):
     train_cur = np.load('../img_array_train_6k_%d.npy' %i)
     #stacking all the current train data in a vector - before filtering brain "sides"
     train_tmp = np.vstack((train_tmp, train_cur))
train_cur = np.load('../img_array_train_6k_%d.npy' %(i+1))[:52]#[:32]
train_tmp = np.vstack((train_tmp, train_cur))

print("train_tmp.shape is ",train_tmp.shape)

#train_all = []
#train_full_scan_tmp = []
#for i in range(train_tmp.shape[0]): #range(21*6000-1):
#    train_full_scan_tmp.append(train_tmp[i])
#    if ((i+1) % 62 == 0):
#        train_all.append(train_full_scan_tmp)    
#	    train_full_scan_tmp.clear()

train_all_list = []
for i in range(train_tmp.shape[0]): #range(21*6000-1):
    train_all_list.append(train_tmp[i])

train_all = np.asarray(train_all_list)
train_all = train_all.reshape(-1, 96, 96, 62) ####fix it to be 62


demo = pd.read_csv('../adni_demographic_master_kaggle.csv')
# gets all the indices of 0 = train and puts all of their 
# data in trX_subjs
trX_subjs_train = demo[(demo['train_valid_test']==0)]
# puts the diagnosis in trY 0/1/2
trY = np.asarray(trX_subjs_train.diagnosis)

trY_all = []
for n in range(len(trY)):		# len(trY) = 2109
    #print("n is " ,n)
    #for i in range(62): # duplicating diagnosis for each slice
    trY_all.append(trY[n])

print("Sanity check:")
#print("len(train_all) is " ,len(train_all))
#print("trY_all.shape is " ,trY_all.shape)
print("trY_all is " ,len(trY_all))
print ("-----loaded .npy files-----")

# #######################################
            #Building network#
# #######################################
x = tf.placeholder(tf.float32, shape=[None, 96 , 96, 62, 1])
y_ = tf.placeholder(tf.float32, shape = [None, 3])		
x_image=x


##### first convolutional later #####
W_conv1 = weight_variable([5 ,5 ,5 , 1, 32], "W_conv1")
# 5X5X5 receptive field ,1 input channel, 32 feature maps
b_conv1 = bias_variable([32], "b_conv1")
#32 feature maps - bias

h_conv1 = tf.nn.relu(conv3d(x_image, W_conv1,"first layer") + b_conv1)
# 96X96X62 -> 0-padding -> 100X100X66 -> conv -> 96X96X33X32
h_pool1 = max_pool_2x2(h_conv1,"first layer")
# input: 96X96X32 -> max_pool -> 48X48X16?X32

##### second convolutional layer #####
W_conv2 = weight_variable([5, 5, 5, 32, 64], "W_conv2")
# 5X5 receptive field ,32 input channel, 64 feature maps
b_conv2 = bias_variable([64],"b_conv2")
#64 feature maps - bias

h_conv2 = tf.nn.relu(conv3d(h_pool1, W_conv2,"second layer") + b_conv2)
# input is 48X48X16X32 ---->  48X48X16X64
h_pool2 = max_pool_2x2(h_conv2,"second layer")
# input is 48X48X16X64 ----> 24X24X8X64
'''
##### third convolutional layer #####
W_conv3 = weight_variable([3, 3, 3, 64, 64], "W_conv3")
# 3X3 receptive field ,64 input channel, 64 feature maps
b_conv3 = bias_variable([64],"b_conv3")
#64 feature maps - bias

h_conv3 = tf.nn.relu(conv3d(h_pool2, W_conv3,"third layer") + b_conv3)
# input is 24X24X8X64 ---->  24X24X8X64
h_pool3 = max_pool_2x2(h_conv3,"third layer")
# input is 24X24X8X64 ----> 12X12X4X64
'''
##### fourth convolutional layer #####
W_conv4 = weight_variable([3, 3, 3, 64, 64], "W_conv4")
# 3X3 receptive field ,64 input channel, 128 feature maps
b_conv4 = bias_variable([64],"b_conv4")
#128 feature maps - bias

h_conv4 = tf.nn.relu(conv3d(h_pool2, W_conv4,"fourth layer") + b_conv4)
# input is 12X12X4X64 ---->  12X12X4X128
h_pool4 = max_pool_2x2(h_conv4,"fourth layer")
# input is 12X12X4X128 ----> 6X6X2128




# first fully connected layer
W_fc1 = weight_variable([6*6*4*64, 64],"W_fc1")
b_fc1 = bias_variable([64],"b_fc1")

h_pool4_flat = tf.reshape(h_pool4, [-1, 6*6*4*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
'''
# second fully connected layer
W_fc2 = weight_variable([64, 64],"W_fc2")
b_fc2 = bias_variable([64],"b_fc2")

#h_pool2_flat = tf.reshape(h_pool2, [-1, 6 * 6 * 128])
h_fc2 = tf.nn.relu(tf.matmul(h_fc1, W_fc2) + b_fc2)
'''
# dropout
#keep_prob = tf.placeholder(tf.float32)
#h_fc_drop = tf.nn.dropout(h_fc2, keep_prob)

# softmax
W_fc3 = weight_variable([64, 3],"W_fc3")
b_fc3 = bias_variable([3],"b_fc3")

y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc3) + b_fc3)

print ("-----CNN architecture built-----")

#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.clip_by_value(y_conv, 1e-10,1.0)), reduction_indices=[1]))
#cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
cross_entropy = tf.reduce_mean(
      tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#tf.summary.scalar('cross_entropy', cross_entropy)


train_step = tf.train.AdamOptimizer(1e-6).minimize(cross_entropy)
# with tf.name_scope('train'):
#     #train_step = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cross_entropy)
#train_step = tf.train.GradientDescentOptimizer(1e-3).minimize(cross_entropy)

#with tf.name_scope('accuracy'):
#    with tf.name_scope('correct_prediction'):
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
#    with tf.name_scope('accuracy'):
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 #   tf.summary.scalar('accuracy', accuracy)


##correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
# when the y_conv is equal to given y_ then correct_prediction==1
# when the y_conv prediction is wrong, correct_prediction==1
##accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
# we want to minimize the ~accuracy~ parameter, when it is zero - all predictions are correct
saver = tf.train.Saver()
sess = tf.Session()
#merged = tf.summary.merge_all()
#train_writer = tf.summary.FileWriter(DIR,sess.graph)
init = tf.global_variables_initializer()
sess.run(init)

print ("started session")


#restoring older weights if such exists

ckpt = tf.train.get_checkpoint_state(DIR)
print(ckpt)
if ckpt:
    print(ckpt.model_checkpoint_path)
if ckpt and ckpt.model_checkpoint_path:
    saver.restore(sess, ckpt.model_checkpoint_path)


train_lables = []
train_stack = []
for i in range(train_all.shape[0]):
    label = [0,0,0]
    label[trY_all[i]-1] = 1
    label = np.asarray(label)
    train_lables.append(label)
    #a = train_all[i].reshape(96,96,1)
    a = np.expand_dims(train_all[i],96)
    train_stack.append(a)
####################### valid set loading and modeling to CNN size #############


valid_tmp = np.load('../img_array_valid_6k_1.npy')
for i in range(2,6):
    valid_cur = np.load('../img_array_valid_6k_%d.npy' %i)
    valid_tmp = np.vstack((valid_tmp, valid_cur))

# print("sanity check: valid_tmp size suppose to be 26,970, it is: ",valid_tmp.shape)
#
valid_allX_trim_list = []
for i in range (valid_tmp.shape[0]):
    #if ((i%62)>19 and (i%62)<40):
    valid_allX_trim_list.append(valid_tmp[i])

valid_allX_trim = np.asarray(valid_allX_trim_list)
valid_allX_trim = valid_allX_trim.reshape(-1, 96, 96, 62)


demo = pd.read_csv('../adni_demographic_master_kaggle.csv')
validY_subjs = demo[(demo['train_valid_test']==1)]
validY_before_dup = np.asarray(validY_subjs.diagnosis)

validY_trim_after_dup = []
for n in range(len(validY_before_dup)):
    #for i in range(20): # duplicating diagnosis for each slice
    validY_trim_after_dup.append(validY_before_dup[n])
validY_trim_after_dup = np.asarray(validY_trim_after_dup)
# validY_trim_after_dup holds all labels for valid set - duplicated 40 times for each subject



validY_stack = []
validX_stack = []
for i in range(len(validY_trim_after_dup)):
    label = [0,0,0]
    label[validY_trim_after_dup[i]-1] = 1
    label = np.asarray(label)
    validY_stack.append(label)
#		a = valid_allX_trim[i].reshape(96,96,1)
    a = np.expand_dims(valid_allX_trim[i],96)
    validX_stack.append(a)

##################################################################################
                            ## print sizes of vectors ##
##################################################################################
print("size of train_all should  == to train_stack")
#print("train_all ", len(train_all))
print("train_stack  ", len(train_stack))
print("train_lables.shape  ", len(train_lables))
print("validX_stack  ", len(validX_stack))
print("validY_stack  ", len(validY_stack))


batch_size = 16 #32
batch_valid_accuracy = 0
batch_train_accuracy = 0
accuracy_sum = 0
valid_accuracy = 0
#print("len(train_all)/batch_size) is " ,math.floor(len(train_all)/batch_size))

#batch_num = int(math.floor(len(train_all)/batch_size))
batch_num = int(math.floor(train_all.shape[0]/batch_size))
batch_num_valid = int(math.floor(valid_allX_trim.shape[0]/batch_size))
print("batch_num = %d" %(batch_num))

for train_iter in range(25000):
    print("train_iter number: ", train_iter)
    for i in range(batch_num):
        Xtr_train, Ytr_train = next_batch(batch_size, train_stack, train_lables)
        sess.run(train_step, feed_dict={x:Xtr_train, y_:Ytr_train}) # , keep_prob:0.5
        #sess.run(train_step, feed_dict={x:train_stack[batch_size*i:batch_size*i+(batch_size-1)], y_:train_lables[batch_size*i:batch_size*i+(batch_size-1)], keep_prob:0.5})
        if i%50 == 0:
            #Xtr_validate, Ytr_validate = next_batch(batch_size, validX_stack, validY_stack)
            train_accuracy = sess.run(accuracy,feed_dict={x:Xtr_train, y_:Ytr_train}) # , keep_prob:0.5
            #valid_accuracy = sess.run(accuracy,feed_dict={x:Xtr_validate, y_:Ytr_validate})
             
            valid_accuracy = 0
            accuracy_sum = 0

            for j in range(batch_num_valid):
                Xtr_validate, Ytr_validate = next_batch_valid(batch_size, validX_stack, validY_stack, j)
                valid_accuracy_cur = sess.run(accuracy, feed_dict={x: Xtr_validate, y_: Ytr_validate})
                accuracy_sum = accuracy_sum + valid_accuracy_cur

            valid_accuracy = accuracy_sum/(batch_num_valid+1)


           #batch_valid_accuracy = batch_valid_accuracy + valid_accuracy*100
            print("step %d,train accuracy, %f" %(i,train_accuracy*100))
            print("step %d, validation accuracy, %f" %(i,valid_accuracy*100))
            print("cross_entropy = ",sess.run(cross_entropy,feed_dict={x:Xtr_train, y_:Ytr_train})) # , keep_prob:0.5
            #print("correct_prediction = ",sess.run(correct_prediction,feed_dict={x:Xtr_train, y_:Ytr_train})) 
            ###print("step %d,valid accuracy, %f" %(i,valid_accuracy*100))
    #print("iteration valid accuracy: %f" %(batch_valid_accuracy/(51)))
    batch_valid_accuracy = 0
    #summary, _ = sess.run([merged, train_step], feed_dict={x:Xtr_train, y_:Ytr_train, keep_prob:0.5})
    #train_writer.add_summary(summary, i)
    a = saver.save(sess, DIR+"mode_3d_with_MCI.ckpt")
    ##print("saved information is:", a)
    ##print("saved session weights to file")





train_writer.close()
#test_writer.close()

