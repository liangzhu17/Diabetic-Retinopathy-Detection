
# set up
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
from packaging import version

import tensorflow as tf
from tensorflow import keras

import numpy as np

print("TensorFlow version: ", tf.__version__)
assert version.parse(tf.__version__).release[0] >= 2, \
    "This notebook requires TensorFlow 2.0 or above."
    
# create log data
logdir = "./log"
tensorboard_callback = k.callbacks.TensorBoard(log_dir=logdir)

# build and train the model
conv_base = k.applications.xception.Xception(weights='imagenet',
                                             include_top=False,        # excluding the last dense layer
                                             input_shape=(256,256,3),
                                             pooling='avg')

model = k.Sequential()
model.add(conv_base)
model.add(l.Dense(512,activation='relu'))
model.add(l.Dense(1,activation='sigmoid'))
model.summary()

conv_base.train = True
fine_tune_at = -33
for layer in conv_base.layers[:fine_tune_at]:
  l.trainable = False

model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.0005/10),
              metrics=['accuracy'])

epochs = 5
fine_tune_epochs = 5 
total_epochs = epochs + fine_tune_epochs

history = model.fit(   
    train_ds,
    validation_data = val_ds,
    shuffle = True,
    steps_per_epoch = 70,
    validation_steps = 2,
    epochs = 3,
    callbacks=[tensorboard_callback]
)

# launch tensorboard
!pip uninstall tb-nightly tensorboard tensorflow tensorflow-estimator tensorflow-gpu tf-estimator-nightly
!pip install tensorflow-gpu
!pip3 install --upgrade grpcio

!wget https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip
!unzip ngrok-stable-linux-amd64.zip

LOG_DIR = './log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOG_DIR)
)
 
get_ipython().system_raw('./ngrok http 6006 &')
 
! curl -s http://localhost:4040/api/tunnels | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
