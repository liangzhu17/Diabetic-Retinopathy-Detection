import tensorflow as tf
import os
import glob
import numpy as np
import numpy.random as npr
import pandas as pd
import tensorflow.keras as k
import tensorflow.keras.layers as l
import matplotlib.pyplot as plt
import random
import tensorboard
import datetime

from tensorboard import notebook

# define an optimizer
# opt = k.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
opt = k.optimizers.Adam(learning_rate=0.001)
# calculate loss and accuracy
train_loss = tf.keras.metrics.Mean()
train_accuracy = tf.keras.metrics.BinaryAccuracy()
val_loss = tf.keras.metrics.Mean()


# using grading descent method to define training step
@tf.function
def train_step(model, images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images)
        loss = tf.keras.losses.binary_crossentropy(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


# create checkpoints
import datetime

opt = tf.keras.optimizers.Adam(0.001)
ckpt = tf.train.Checkpoint(step=tf.Variable(1), optimizer=opt, net=model)
manager = tf.train.CheckpointManager(ckpt, './tf_ckpts', max_to_keep=4)


# create tensorboards
# train_summary_writer, val_summary_writer = tensorboard_setup()

def train_and_checkpoint(net, manager):
    ckpt.restore(manager.latest_checkpoint)
    if manager.latest_checkpoint:
        print("Restored from {}".format(manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")


# Commented out IPython magic to ensure Python compatibility.
# load tensorboard

# %load_ext tensorboard

# custom defined training step
train_iters = 0

val_acc_metric = k.metrics.BinaryAccuracy()
current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
train_log_dir = 'logs/gradient_tape/' + current_time + '/train'
val_log_dir = 'logs/gradient_tape/' + current_time + '/val'
pro_log_dir = 'logs/profile/' + current_time
train_summary_writer = tf.summary.create_file_writer(train_log_dir)
val_summary_writer = tf.summary.create_file_writer(val_log_dir)
pro_summary_writer = tf.summary.create_file_writer(pro_log_dir)

for t_i, t_l in train_ds.take(-1):

    try:
        tf.enable_eager_execution()
    except Exception:
        pass
    train_step(model, t_i, t_l)
    ckpt.step.assign_add(1)
    if int(ckpt.step) % 10 == 0:
        save_path = manager.save()
        model.save('my_model.h5')
        print("Saved checkpoint for step {}: {}".format(int(ckpt.step), save_path))

    train_iters += 1
    if train_iters % 10 == 0:
        for v_i, v_l in val_ds:
            v_pred = model(v_i)
            # Update val metrics
            val_acc_metric(v_l, v_pred)
            v_loss = tf.keras.losses.binary_crossentropy(v_l, v_pred)
            v_loss = tf.math.reduce_mean(v_loss)
            # val_loss = tf.reduce_mean(tf.square(v_l-v_pred))
            with val_summary_writer.as_default():
                tf.summary.scalar('val_loss', v_loss, step=train_iters)
                tf.summary.scalar('val_accuracy', val_acc_metric.result(), step=train_iters)

        print('Validation acc: %s' % (float(val_acc_metric.result() * 100)))
        print('Validation loss: %s' % (float(v_loss)))
        val_acc_metric.reset_states()

    with train_summary_writer.as_default():
        tf.summary.scalar('train_loss', train_loss.result(), step=train_iters)
        tf.summary.scalar('train_accuracy', train_accuracy.result(), step=train_iters)

    if train_iters == 1600:
        break

    template = 'train_iters {}, Loss: {}, Accuracy: {}'

    print(template.format(train_iters + 1,
                          train_loss.result(),
                          train_accuracy.result() * 100))

# calculate test accuracy

train_iters = 0
test_acc_metric = k.metrics.BinaryAccuracy()

for i, l in test_ds:
    pred = model(i)
    test_acc_metric(l, pred)  # Update val metrics

test_acc = test_acc_metric.result()
test_acc_metric.reset_states()
print('test acc: %s' % (float(test_acc)))
