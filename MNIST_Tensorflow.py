# -*- coding: utf-8 -*-
"""
Created on Tue Apr  4 14:48:51 2023

@author: Nautilus
"""
import os
#import numpy as np
import tensorflow as tf
print("TensorFlow version:", tf.__version__)
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn import preprocessing

# Set the working directory
path='F:/Semester 1 Spring 2023/Machine Learning for CVEN/Project_4'
os.chdir(path)

#Load MNIST dataset
#mnist = tf.keras.datasets.mnist

#set up the training and dataset
(x_train, y_train), (x_test, y_test) = mnist.load_data(path=path+'/mnist.npz')
x_train, x_test = x_train / 255.0, x_test / 255.0 ##Scale the values of image to a range 0 to 1

#Build a machine learning model
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10)
])

predictions = model(x_train[:1]).numpy()
predictions

tf.nn.softmax(predictions).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(y_train[:1], predictions).numpy()

#Compile the model with loss function, optimizer, metrics
model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

# Using a custom callback to plot the total time taken to fit certain epochs.
class timecallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.times = []
        self.epochs = []
        # use this value as reference to calculate cummulative time taken
        self.timetaken = tf.timestamp()
    def on_epoch_end(self,epoch,logs = {}):
        self.times.append(tf.timestamp() - self.timetaken)
        self.timetaken = tf.timestamp()
        self.epochs.append(epoch)
    def on_train_end(self,logs = {}):
        print('Total training time:', sum(self.times))
        plt.figure(figsize=(6,4))
        plt.xlabel('Epoch')
        plt.ylabel('Training time per epoch (s)')
        plt.plot(self.epochs, self.times)
        plt.grid()
        
timetaken = timecallback()

history = model.fit(x_train, y_train, epochs=300, batch_size=64,
                     # Suppress logging.
                     #verbose=0,
                     # use callback to plot training time
                     callbacks = [timetaken])

hist = pd.DataFrame(history.history)
hist['epoch'] = history.epoch
print(hist.head(5))
print(hist.tail(5))

plt.figure(figsize=(6,4))

plt.plot(hist['epoch'], hist['loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()        
#train and fit the model to training data
#model.fit(x_train, y_train, epochs=5)

#Evaluate the accuracy
model.evaluate(x_test,  y_test, verbose=2)

probability_model = tf.keras.Sequential([
  model,
  tf.keras.layers.Softmax()
])

#make prediction
probability_model(x_test[:5])

