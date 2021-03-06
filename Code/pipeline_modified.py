# -*- coding: utf-8 -*-
"""pipeline_modified.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1wzVO-v-VBHi9eEGXFO68UOBn5kQbkSy6
"""

!pip3 install tensorflow-gpu==2.0.0-beta0

import tensorflow as tf
import os
import glob
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow.keras as k
import tensorflow.keras.layers as l
import matplotlib.pyplot as plt
# import tensorflow_addons as tfa
import random
import cv2
import tensorboard
import datetime

from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model
from keras import optimizers
from tensorboard import notebook
from sklearn.metrics import confusion_matrix

# create connection with colab
from google.colab import drive
drive.mount('/content/gdrive')

buffersize = 50
batchsize = 30
N_samples = 413
N_trainingsamples = 383
N_validationsamples = 30
N_testsamples = 103
learning_rate = 0.001

"""get the filenames of images in training set"""
name_path = "/content/gdrive/My Drive/Masterlab/a. Training Set"
file_dir = os.listdir(name_path)


"""check the order of the filenames"""
print(sorted(file_dir))  
 

N_prefetch = 8     # Define the number of data to be prefetched for training
N_parallel_iteration = 4  


# load the csv file for labels
csv_path_train = "/content/gdrive/My Drive/Masterlab/a. IDRiD_Disease Grading_Training Labels.csv"
files_csv_train = pd.read_csv(csv_path, usecols=[1])


"""Create training labels and validation labels together in one-hot coding form, 
   they will be seperated later"""
def create_label(csv, sample_num):
  labels = np.zeros(shape=(sample_num, 2))   
  csv_tensor = tf.convert_to_tensor(csv.values, dtype=tf.int32)
  csv_tensor = tf.map_fn(lambda x: 1 if x > 1 else 0, csv.values)
  for i in range(sample_num):

    if csv_tensor[i] == 1:
        labels[i][0] = 1
    else:
        labels[i][1] = 1

  return labels


labels = create_label(files_csv_train, N_samples)   


# build dataset of images(Training and validation set)

def load_file_names():
    files = glob.glob("/content/gdrive/My Drive/Masterlab/a. Training Set/*.jpg")
    return files


files = sorted(load_file_names())

"""define data augmentation functions"""

"""flip a image upsidedown"""
@tf.function
def flip1(img):
    img_flipped = tf.image.random_flip_up_down(img)
    img = tf.cast(img_flipped, tf.float32) / 255.0  # normalise the image after flippig
    return img


"""flip a image left and right"""
def flip2(img):
    img_flipped = tf.image.random_flip_left_right(img)
    img = tf.cast(img_flipped, tf.float32) / 255.0    # normalise the image after flippig
    return img


"""rotate a image in an random selected angle from 0 to 20"""
def rotate(img):
    angles = tf.random.uniform([], minval=0, maxval=20, dtype=tf.dtypes.float32)
    img = tfa.image.rotate(img, angles, interpolation='NEAREST', name=None)
    img = tf.cast(img, tf.float32) / 255.0
    return img


"""zoom a image """
def zoom(img):
    scales = list(np.arange(0.8, 1.0, 0.01))  
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    """Create different crops for an image and return a random crop"""
    def random_crop(img):
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(256,256))
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]  

    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    """Only apply cropping 50% of the time"""
    return tf.cond(choice < 0.5, lambda: img, lambda: random_crop(img))


"""Rotate a image 90 degree"""
def rot90(img):
    img = tf.image.rot90(img, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    img = tf.cast(img, tf.float32) / 255.0
    return img


"""Create summary of all augmentations"""
augmentations = [flip1, flip2, rot90, zoom]


"""The function get one filename and returns an image"""
def parse_function(files):
    image_string = tf.io.read_file(files)
    image_decoded = tf.io.decode_jpeg(image_string)
    image_resized = tf.image.resize_with_pad(image_decoded, 256, 256)
    return image_resized


"""Build training dataset and apply online augmentation"""
def build_train_ds(files, labels, batchsize):
    ds_x = tf.data.Dataset.from_tensor_slices(files)
    ds_x = ds_x.map(parse_function, N_parallel_iteration)
    # Apply the augmentation, run 4 jobs in parallel.
    # Apply to the training dataset
    for f in augmentations:   
      ds_x = ds_x.map(f)
    # Make sure that the values are still in [0, 1]
    ds_x = ds_x.map(lambda x: tf.clip_by_value(x, 0, 1))
    ds_y = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((ds_x,ds_y))
    ds = ds.shuffle(380).batch(batchsize).repeat(-1).prefetch(N_prefetch)
    return ds


"""Build validation dataset without augmentation"""
def build_val_ds(files, labels, batchsize):
    ds_x = tf.data.Dataset.from_tensor_slices(files)
    ds_x = ds_x.map(parse_function)
    ds_y = tf.data.Dataset.from_tensor_slices(labels)
    ds = tf.data.Dataset.zip((ds_x,ds_y))
    ds = ds.shuffle(20).batch(batchsize).prefetch(N_prefetch)
    return ds


"""Shuffle the data before split them to training and validation sets"""
shuffle_idx = np.arange(0, N_samples)
np.random.shuffle(shuffle_idx) 
files = [files[i] for i in shuffle_idx]       # shuffle index of filenames, shortens time
labels = [labels[i] for i in shuffle_idx]


"""Build total dataset(training set and validtion set)"""
train_ds = build_train_ds(files[0:N_trainingsamples], labels[0:N_trainingsamples], batchsize)
val_ds = build_val_ds(files[N_trainingsamples:N_samples], labels[N_trainingsamples:N_samples], batchsize)

"""build test dataset of images"""

def load_testfile_names():
    files = glob.glob(
        "/content/gdrive/My Drive/Masterlab/b. Testing Set/*.jpg")
    return files


test_img_files = load_testfile_names()


def build_test_img_ds(input_file):
  img_list_test = []
  for file in sorted(input_file):
    image_string = tf.io.read_file(file)
    image_decoded = tf.io.decode_image(image_string)
    image_resized = tf.image.resize_with_pad(image_decoded, 256, 256)
    img = tf.cast(image_resized, tf.float32) / 255.0
    img_list_test.append(img)

    img_tensor_test = tf.convert_to_tensor(img_list_test, dtype=tf.float32)
    img_test_ds = tf.data.Dataset.from_tensor_slices(img_tensor_test)

  return img_test_ds


# build dataset of labels(testing set)

csv_path_test = "/content/gdrive/My Drive/Masterlab/b. IDRiD_Disease Grading_Testing Labels.csv"
files_csv_test = pd.read_csv(csv_path_test, usecols=[1])


"""build total dataset(testing set)"""
def build_test_ds(input_file, batchsize):
  img_ds = build_test_img_ds(input_file)

  labels_test = create_label(files_csv_test, N_testsamples)   
  label_ds_test = tf.data.Dataset.from_tensor_slices(labels_test)
  ds = tf.data.Dataset.zip((img_ds, label_ds_test)).batch(batchsize)
  return ds

  
test_ds = build_test_ds(test_img_files, batchsize)