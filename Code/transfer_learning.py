
# load base model----Xception
conv_base = k.applications.xception.Xception(weights='imagenet',
                                             include_top=False,         # excluding the last optional fully connected layer
                                             input_shape=(256,256,3),
                                             pooling='avg')

# Add two dense layers
model = k.Sequential()
model.add(conv_base)
model.add(l.Dense(512,activation='relu'))
model.add(l.Dense(1,activation='sigmoid'))
model.summary()

# the first 100 layers: set as untrainable
# the last 33 layers: set as fine tuning layers
conv_base.train = True
fine_tune_at = -33
for layer in conv_base.layers[:fine_tune_at]:
  l.trainable = False
  
# compile the model
model.compile(loss='binary_crossentropy',
              optimizer=tf.keras.optimizers.Adam(lr=0.001/20), # learning rate is 1/20 of self defined model
              metrics=['accuracy'])

# set epochs
epochs = 5
fine_tune_epochs = 5
total_epochs = epochs + fine_tune_epochs

# use fit method to train the model
history = model.fit(   
    train_ds,
    validation_data = val_ds,
    shuffle = True,
    steps_per_epoch = 70,
    validation_steps = 10,
    epochs = total_epochs,
    callbacks=[tensorboard_callback]
)

# plot the training and validation accuracy
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label = 'val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')

# calculate test accuracy
test_loss, test_acc = model.evaluate(test_ds, verbose=2)
print(test_acc)
