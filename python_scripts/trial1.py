# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 11:15:05 2020

@author: Celia
"""

import numpy as np
import keras
import mnist

#Building the Model
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical

# The first time you run this might be a bit slow, since the
# mnist package has to download and cache the data.
train_images = mnist.train_images()
train_labels = mnist.train_labels()
test_images = mnist.test_images()
test_labels = mnist.test_labels()

print(train_images.shape)
print(train_labels.shape)

# Normalize the images.
train_images = (train_images / 255) - 0.5
test_images = (test_images / 255) - 0.5

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

print(train_images.shape) # (60000, 784)
print(test_images.shape)  # (10000, 784)


#Building the model
# WIP
model = Sequential([
  # layers...
])

# Still a WIP
#model = Sequential([
  #Dense(64, activation='relu'),
  #Dense(64, activation='relu'),
  #Dense(10, activation='softmax'),
#])


#Defiining the Dense Layer
model = Sequential([
  Dense(64, activation='relu', input_shape=(784,)),
  Dense(64, activation='relu'),
  Dense(10, activation='softmax'),
])

#Compiling the model
model.compile(
  optimizer='adam',
  loss='categorical_crossentropy',
  metrics=['accuracy'],
)


#Training the model
model.fit(
  train_images,# training data
  to_categorical(train_labels), # training targets
  epochs=5,
  batch_size=32,
)


model.evaluate(
  test_images,
  to_categorical(test_labels)
)


# Save the model to disk.
model.save_weights('model.h5')

# Load the model from disk later using:
# model.load_weights('model.h5')

# Predict on the first 5 test images.
predictions = model.predict(test_images[:5])

# Print our model's predictions.
print(np.argmax(predictions, axis=1)) # [7, 2, 1, 0, 4]

# Check our predictions against the ground truths.
print(test_labels[:5]) # [7, 2, 1, 0, 4]