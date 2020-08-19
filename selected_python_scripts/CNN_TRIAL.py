#!/usr/bin/env python
# coding: utf-8
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import Sequential
from keras.models import Input
from keras.layers import Dense
from keras.layers import InputLayer
from keras.layers import Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers import BatchNormalization
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import scipy.stats
import tensorflow as tf
import time
import graphviz
import pydot
import os

print(keras.__version__)

# remove tf warning messages
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import tensorflow as tf


print('Loading dataset...')
#Import the files
train_df= pd.read_csv("merge_table_debug_finals_alltogether_runs.csv", sep=" ",  header = None) 
print('Shape of data: ' + str(train_df.shape))

print()
print('Example datapoint:')
print(train_df.iloc[0])
print()
## Generate labels
ngenes = 300 #Number of genes (columns?)
n_split=5  #Value of K for K-fold cross-validation10

# reshape so each row is one simulation/ flatten the images??
#train_df= np.reshape(train_df, (3942,ngenes))
#train_df= np.reshape(train_df, (,300))


print('Creating labels...')
# make label array, first half simple (0), second half complex (1)
all_labels = np.full((3998,1), 0)
all_labels[:1999, 0].fill(1)
all_labels[1999: , 0].fill(0)
all_labels

print()
print("Labels generated:")
print(all_labels)
print()


print('Shuffling data...')
## shuffles rows of dataset (and labels)
def shuffle_elements(data, labels):
    labels = pd.DataFrame(labels)
    data['lab'] = labels
    
    data = data.sample(frac=1).reset_index(drop=True)
    return data.iloc[:, 0:300], data['lab']


train_df, all_labels = shuffle_elements(train_df, all_labels)


# # Define model
print('Defining model...')
print()
model = tf.keras.Sequential()
model.add(InputLayer((300,1)))
model.add(Conv1D(filters=10, kernel_size=5, activation='relu', input_shape=(300, 1)))
model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))
model.add(Conv1D(filters=40, kernel_size=20, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))
model.add(Conv1D(filters=60, kernel_size=30, activation='relu'))
model.add(MaxPooling1D(pool_size=2, strides=None, padding="valid"))
model.add(Dense(100, activation='relu'))
#model.add(MaxPooling1D(pool_size=2))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# # Define function to reset model (used for running kfold training)
def shuffle_weights(model, weights=None):
    """Randomly permute the weights in `model`, or the given `weights`.

    This is a fast approximation of re-initializing the weights of a model.

    Assumes weights are distributed independently of the dimensions of the weight tensors
      (i.e., the weights have the same distribution along each dimension).

    :param Model model: Modify the weights of the given model.
    :param list(ndarray) weights: The model's weights will be replaced by a random permutation of these weights.
      If `None`, permute the model's current weights.
    """
    if weights is None:
        weights = model.get_weights()
    weights = [np.random.permutation(w.flat).reshape(w.shape) for w in weights]
    # Faster, but less random: only permutes along the first dimension
    # weights = [np.random.permutation(w) for w in weights]
    model.set_weights(weights)
    return model


# Train

total_accuracy = 0
total_epochs= 90

# Define binary output accuracy: threshold for label: 0.5
tf.keras.metrics.BinaryAccuracy(
    name="binary_accuracy", dtype=None, threshold=0.5
)


#K-fold cross-validation
for train_index,test_index in KFold(n_split).split(train_df):
    
        #print('Train: %s | test: %s' % (train_index, test_index))
        x_train,x_test=train_df.iloc[train_index, :],train_df.iloc[test_index, :]
        y_train,y_test=all_labels[train_index],all_labels[test_index]
        
        
        #x_train.reshape(3200, 300, 1)
        x_train = np.expand_dims(x_train, axis=2)
        x_test = np.expand_dims(x_test, axis=2)
        
        print('Shape batch (size expandeed by one dimension for tensorflow requirements):')
        print(x_train.shape) 
        print(x_test.shape)
              
       
        #Compile the model/ Train the model
        model.compile(loss='binary_crossentropy',
                      optimizer='Adam',
                      metrics=[tf.keras.metrics.BinaryAccuracy()])
        
        model = shuffle_weights(model)
        
        history=model.fit(x_train, y_train,
                          epochs=total_epochs,
                          validation_data = (x_test, y_test), 
                          verbose=1)
                
        (loss, accuracy) = model.evaluate(x_test,y_test)
        total_accuracy = total_accuracy + accuracy
        
        #print(history.history)
        
        #Accuracy"
        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('Model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.ylim([0,1.1])
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.clf()
        
        
            
        # Visualize history
        # Plot history: Loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('Validation loss history')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()



#Calculate average accuracy of 1 complete run through the K-fold cross-validation
final_accuracy = total_accuracy/n_split
print('final_accuracy', final_accuracy)
        #scores[i] = final_accuracy
        #Calculate mean and CI from 10 runs through K-fold
        #(mean, lower, upper) = mean_confidence_interval(scores)
        #Append scores from model with current number of nodes
        #data.append([nodes,mean,lower,upper,(upper-lower)])




