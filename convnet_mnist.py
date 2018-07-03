
# coding: utf-8

# 


from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
from math import pi

data = input_data.read_data_sets('./data/', one_hot = False)
data_X = data.train.images
data_y = data.train.labels


def one_hot_single(y):
    vector = np.zeros((y.shape[0], 9))
    
    for i in range(y.shape[0]):
        if y[i] == 6 or y[i] == 9:
            vector[i][6] = 1.
        else:
            vector[i][y[i]] = 1.
    
    return vector


# 


import tensorflow as tf

n_classes = 9
batch_size = 200
epochs = 50

x = tf.placeholder('float', [None, 28, 28, 1])
y = tf.placeholder('float', [None, 9])


# 


## Parameters
sigma = 1.0

weights_conv1 = tf.Variable(tf.random_normal([5, 5, 1, 32], stddev = sigma)) # First Filters
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

def forward_pass(X, Y):
    
    with tf.name_scope("Convolution_Layer_1"):
        conv1 = tf.nn.conv2d(X, weights_conv1, strides = [1, 1, 1, 1], padding = 'SAME')
        conv1 = tf.nn.relu(conv1)
        conv1 = tf.nn.max_pool(conv1, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.name_scope("Convolution_Layer_2"):
        conv2 = tf.nn.conv2d(conv1, weights_conv2, strides = [1, 1, 1, 1], padding = 'SAME')
        conv2 = tf.nn.relu(conv2)
        conv2 = tf.nn.max_pool(conv2, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME')

    with tf.name_scope("Fully_Connected_Layer_1"):
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
    
    costPred = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y, logits = output))
    
    return [output, costPred]


#


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
    
sess = tf.InteractiveSession()
data_test_images = data.test.images
data_test_images = data_test_images.reshape((data.test.num_examples, 28, 28, 1))
sess.run(tf.global_variables_initializer())

rotated_test_images = sess.run(random_rotation(x), feed_dict = {x : data_test_images})
rotated_test_images_ = rotated_test_images[0 : 1000]
rotated_test_labels = one_hot_single(data.test.labels)
rotated_test_labels_ = rotated_test_labels[0 : 1000]
#
def train(x, y):
    output, costPred = forward_pass(x, y)
    
    opt = tf.train.AdamOptimizer()
    
        num_epochs = 10
        for i in range(10):
            with tf.Session() as sess:
    
            
                sess.run(tf.global_variables_initializer())
                for epoch in range(num_epochs):
                    epoch_loss = 0
                    counter_batch_done = 0
                    for _ in range(10000 // batch_size):
                        epoch_x, epoch_y = data.train.next_batch(batch_size)
                        
                        epoch_x = sess.run(rotation(x), feed_dict = {x : epoch_x.reshape((batch_size, 28, 28, 1))})
                        epoch_y = one_hot_single(epoch_y)
                        grads_and_vars = opt.compute_gradients(costPred)
                        train_opt = opt.apply_gradients(grads_and_vars)
                        _, c = sess.run([train_opt, costPred], feed_dict = {x : epoch_x, y : epoch_y})
                        print(c)
                                            #z_tensor = tf.gradients(costPred, weights_conv1)       
                        #z_val = sess.run(z_tensor, feed_dict = {x : epoch_x, y : epoch_y})
                        #print(z_val)
                        epoch_loss += c
                        print(counter_batch_done, ' batch_done')
                        counter_batch_done = counter_batch_done + 1
                    #print('Epoch ', epoch, 'completed out of ', num_epochs, 'loss : ', epoch_loss)
                        correct = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1))
                        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
                        z1 = accuracy.eval({x : rotated_test_images_, y : rotated_test_labels_})
                        print(z1)
#        T = [sess.run(weights), sess.run(biases)]
    
            
                


#


train(X, Y)


#


#epoch_x, epoch_y = data.train.next_batch(10)
#epoch_y.shape


# 


#batch_size


# 


#from math import pi as PI
#weights['W_conv1'] = tf.reshape(weights['W_conv1'], [-1, 5, 5, 1])
#weights['W_conv1'] = tf.contrib.image.rotate(weights['W_conv1'], 1, interpolation = 'BILINEAR')


# 


#sess.run(weights['W_conv1'])[0, :, :, 0]


# 


import numpy as np
def central_correlation(img_tensor):
    
    m = img_tensor.shape[0]
    
    ksize = 2
    wsize = 2 * ksize + 1
    
    m_ = (m - 2 * ksize) * wsize
    img_tensor_transformed = np.zeros((m_, m_))
    
    #gw = gaussian_window(wsize)
    step = 1
    for i in range(ksize, m - ksize, step):
        for j in range(ksize, m - ksize, step):
            for k in range(-1 * ksize, ksize + 1):
                for l in range(-1 * ksize, ksize + 1):
                    i_ = k + ksize + int((i - ksize) / step) * wsize
                    j_ = l + ksize + int((j - ksize) / step) * wsize
                    diff = img_tensor[i + k, j + l] - img_tensor[i, j]
                    img_tensor_transformed[i_, j_] = np.e ** (-1 * (diff))
    return img_tensor_transformed


#


image = data_X[np.random.randint(0, 50000, 1)[0]].reshape((28, 28))
plt.imshow(image)


# 


image = central_correlation(image)
plt.imshow(image)


#


image

