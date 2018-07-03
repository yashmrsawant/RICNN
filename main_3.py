"""
Created on Fri Jun  8 07:51:55 2018

@author: yashmrsawant
"""


# coding: utf-8

# In[7]:


from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
from math import pi

data = input_data.read_data_sets('./data/', one_hot = False)
#data_X_train = data.train.images

def one_hot_single(y):
    vector = np.zeros((y.shape[0], 9))
    
    for i in range(y.shape[0]):
        if y[i] == 6 or y[i] == 9:
            vector[i][6] = 1.
        else:
            vector[i][y[i]] = 1.
    
    return vector

#data_y_train = one_hot(data.train.labels) # With 6 and 9 treated as same



# In[5]:


batch_size = 128
epochs = 50

X = tf.placeholder('float', [None, 28, 28, 1])
Y = tf.placeholder('float', [None, 9])

## Parameters

sigma = 1.0
weights_conv1_flat = tf.Variable(tf.random_normal([800]))
weights_conv1 = tf.reshape(weights_conv1_flat, [5, 5, 1, 32]) # First Filters
#biases_conv1 = tf.Variable(tf.random_normal([32])) 

weights_conv2 = tf.Variable(tf.random_normal([5, 5, 32, 64], stddev = sigma))
#biases_conv2 = tf.Variable(tf.random_normal([64]))

weights_fc1 = tf.Variable(tf.random_normal([7 * 7 * 64, 1024], stddev = sigma))
biases_fc1 = tf.Variable(tf.random_normal([1024], stddev = sigma))

#weights_fc2 = tf.Variable(tf.random_normal([1024, 700], stddev = sigma))
#biases_fc2 = tf.Variable(tf.random_normal([700], stddev = sigma))

weights_nca1 = tf.Variable(tf.random_normal([1024, 700], stddev = sigma))
biases_nca1 = tf.Variable(tf.random_normal([700], stddev = sigma))

weights_nca2 = tf.Variable(tf.random_normal([700, 300], stddev = sigma))
biases_nca2 = tf.Variable(tf.random_normal([300], stddev = sigma))

weights_nca3 = tf.Variable(tf.random_normal([300, 70], stddev = sigma))
biases_nca3 = tf.Variable(tf.random_normal([70], stddev = sigma))
# For MNIST, considering 6 and 9 as same softmax output
weights_softmax = tf.Variable(tf.random_normal([70, 9], stddev = sigma)) 
biases_softmax = tf.Variable(tf.random_normal([9], stddev = sigma))


# In[3]:

def forward_pass(X, Y, weights_conv1r):
    
    with tf.name_scope("Convolution_Layer_1"):
        conv1 = tf.nn.conv2d(X, weights_conv1r, strides = [1, 1, 1, 1], padding = 'SAME')
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.name_scope("Convolution_Layer_2"):
        conv2 = tf.nn.conv2d(conv1, weights_conv2, strides = [1, 1, 1, 1], padding = 'SAME')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.name_scope("Fully_Connected_Layer"):
        fc1 = tf.reshape(conv2, [-1, 7 * 7 * 64])
        fc1 = tf.matmul(fc1, weights_fc1) + biases_fc1
        fc1 = tf.nn.relu(fc1)

    with tf.name_scope("NCA_Subspace_Layer_1"):
        nca1 = tf.matmul(fc1, weights_nca1) + biases_nca1
        nca1 = tf.nn.relu(nca1)
    
    with tf.name_scope("NCA_Subspace_Layer_2"):
        nca2 = tf.matmul(nca1, weights_nca2) + biases_nca2
        nca2 = tf.nn.relu(nca2)
    with tf.name_scope("NCA_Subspace_Layer_3"):
        nca3 = tf.matmul(nca2, weights_nca3) + biases_nca3
        nca3 = tf.nn.relu(nca3)        
    with tf.name_scope("Output_Softmax_Layer"):
        output = tf.matmul(nca3, weights_softmax) + biases_softmax
    
    costPred = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = Y, logits = output))
    
    return [fc1, nca3, output, costPred]


# In[80]: Rotational Invariant Filter Objective
    
def rotate_i(weights, i):
    return tf.reshape(tf.contrib.image.rotate(\
            tf.reshape(weights, [32, 5, 5, 1]), pi / 4. * i, interpolation = 'BILINEAR'), [5, 5, 1, 32])

weights_conv1 = {
        'weights_conv10' : tf.reshape(weights_conv1_flat, [5, 5, 1, 32]),
        'weights_conv11' : rotate_i(weights_conv1_flat, 1),
        'weights_conv12' : rotate_i(weights_conv1_flat, 2),
        'weights_conv13' : rotate_i(weights_conv1_flat, 3),
        'weights_conv14' : rotate_i(weights_conv1_flat, 4),
        'weights_conv15' : rotate_i(weights_conv1_flat, 5),
        'weights_conv16' : rotate_i(weights_conv1_flat, 6),
        'weights_conv17' : rotate_i(weights_conv1_flat, 7)
}

fc1 = {'fc10' : None, 'fc11' : None, 'fc12' : None, 'fc13' : None, 'fc14' : None, 
                  'fc15' : None, 'fc16' : None, 'fc17' : None}

fc1['fc10'], nca3, output, costPred = forward_pass(X, Y, weights_conv1['weights_conv10'])
    
for i in range(1, 8): # Orientations
    fc1['fc1' + str(i)], _, _, _ = forward_pass(X, Y, weights_conv1['weights_conv1' + str(i)])

rif_obj = tf.constant([0.])
for i in range(8):
    for j in range(i + 1, 8):
        rif_obj = rif_obj + tf.squared_difference((fc1['fc1' + str(i)]), \
                                                  (fc1['fc1' + str(j)]), \
                                                  name = "RIF_Objective" + str(i) + str(j))
rif_obj = tf.nn.l2_loss(rif_obj)

# In[8]:

def rotation(X):
    
    angles = pi / 4 * np.random.randint(0, 8, batch_size)
    Z = tf.contrib.image.rotate(X, angles, interpolation = 'BILINEAR')
    
    return Z

def random_rotation(X):
    angles = 4 * pi * np.random.random(10000) - pi
    Z = tf.contrib.image.rotate(X, angles, interpolation = 'BILINEAR')
    
    return Z
#
#
data_test_images = data.test.images
data_test_images = data_test_images.reshape((data.test.num_examples, 28, 28, 1))
    
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    rotated_test_images = sess.run(random_rotation(X), feed_dict = {X : data_test_images})
    sess.close()    
#rotated_test_images_ = data_test_images[0 : 1000]
rotated_test_images_ = rotated_test_images[0 : 1000]
rotated_test_labels = one_hot_single(data.test.labels)
rotated_test_labels_ = rotated_test_labels[0 : 1000]
#


# In[20]:


_, _, output, costPred = forward_pass(X, Y, weights_conv1['weights_conv10'])
opt = tf.train.AdamOptimizer()
sess = tf.InteractiveSession()
saver = tf.train.Saver()
num_epochs = 10
sess.close()
# In[5]    
with tf.Session() as sess:            
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_all_variables())
        for epoch in range(num_epochs):
            epoch_loss = 0
            counter_batch_done = 0
            for _ in range(10000 // batch_size):
                epoch_x, epoch_y = data.train.next_batch(batch_size)
                #epoch_x = sess.run(rotation(X), feed_dict = {X : epoch_x.reshape((batch_size, 28, 28, 1))})
                epoch_y = one_hot_single(epoch_y)
                epoch_x = epoch_x.reshape((batch_size, 28, 28, 1))
                grads_and_vars = opt.compute_gradients(costPred)
                train_opt = opt.apply_gradients(grads_and_vars)
                
                grads_and_vars_rif = opt.compute_gradients(rif_obj, var_list = weights_conv1_flat)
                train_opt_rif = opt.apply_gradients(grads_and_vars_rif)               
                _, c = sess.run([train_opt, costPred], feed_dict = {X : epoch_x, Y : epoch_y})
                
                print(c)
                _, c = sess.run([train_opt_rif, rif_obj], feed_dict = {X : epoch_x, Y : epoch_y})
                print(c)
                #z_tensor = tf.gradients(costPred, weights_conv1)       
                #z_val = sess.run(z_tensor, feed_dict = {x : epoch_x, y : epoch_y})
                #print(z_val)
                #print(sess.run(weights_conv1_flat))
                epoch_loss += c
                print(counter_batch_done, ' batch_done')
                counter_batch_done = counter_batch_done + 1
                #print('Epoch ', epoch, 'completed out of ', num_epochs, 'loss : ', epoch_loss)
                correct = tf.equal(tf.argmax(output, 1), tf.argmax(Y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                z1 = accuracy.eval({X : rotated_test_images_, Y : rotated_test_labels_})
                print(z1)
            saver.save(sess, './tmp/model' + str(epoch) + '.ckpt')
            


# In[18]:



