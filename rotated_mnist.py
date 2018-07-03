#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 21:20:08 2018

@author: yashmrsawant
"""

path = '/home/yashmrsawant/Documents/data/'


from os import listdir

import tensorflow as tf
import numpy as np
from skimage import io
from math import pi
import matplotlib.pyplot as plot


files = listdir(path + 'images_training_rev1/')

count = 0
for file in files:
    file_path = path + 'images_training_rev1/' + file
    
    image = io.imread(file_path, as_grey = True)
    image = np.array(image).reshape((image.shape[0], image.shape[1], 1))
    rand_n = np.random.randint(1, 10, size = 1)[0]
    image = sess.run(tf.contrib.image.rotate(tf.image.central_crop(image, 0.08), tf.constant([pi / rand_n * 180.])))
    
    np.savetxt(path + 'processed_images_galaxy_zoo/' + file, \
               image.reshape((image.shape[0], image.shape[1])), delimiter = ',')
    plot.imshow(image.reshape((image.shape[0], image.shape[1])))
    print(str(count) + ' done')
    count = count + 1