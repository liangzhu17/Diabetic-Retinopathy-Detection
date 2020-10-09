import tensorflow as tf
import os
import glob
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow.keras as k
import tensorflow.keras.layers as l
import matplotlib.pyplot as plt
import tensorflow_addons as tfa
import random
import cv2
import tensorboard
import datetime



#build model
#rewrite the model for deep visualization

inputs = k.layers.Input(shape=(256,256,3))
conv0 = k.layers.Conv2D(16, (3,3),  activation='relu')(inputs)
pool0 = k.layers.MaxPooling2D(pool_size=(3, 3))(conv0)
conv1 = k.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(pool0)
pool1 = k.layers.MaxPooling2D(pool_size=(3, 3))(conv1)
drop0 = k.layers.Dropout(0.3)(pool1)
conv2 = k.layers.Conv2D(64, (3, 3), activation='relu', padding="same")(drop0)
pool2 = k.layers.MaxPooling2D(pool_size=(3, 3))(conv2)
conv3 = k.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(pool2)
pool3 = k.layers.MaxPooling2D(pool_size=(3, 3))(conv3)
drop1 = k.layers.Dropout(0.3)(pool3)
conv4 = k.layers.Conv2D(128, (3, 3), activation='relu', padding="same")(drop1)
pool4 = k.layers.MaxPooling2D(pool_size=(3, 3))(conv4)
drop2 = k.layers.Dropout(0.3)(pool4)
flatten0 = k.layers.Flatten()(drop2)
dense0 = k.layers.Dense(512, activation="relu")(flatten0)
drop3 = k.layers.Dropout(0.4)(dense0)
dense1 = k.layers.Dense(128, activation="relu")(drop3)
drop4 = k.layers.Dropout(0.5)(dense1)
dense2 = k.layers.Dense(2, activation="softmax")(drop4)

output = dense2
model = k.models.Model(inputs=inputs, outputs=output)

# compile the model
model.compile(optimizer= 'adam',
              loss='binary_crossentropy',
              metrics=['binary_accuracy'])

model.build((batchsize, 256, 256, 3))
model.summary()
