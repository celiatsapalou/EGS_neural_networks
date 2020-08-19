#!/usr/bin/env python
# coding: utf-8

# In[206]:


import numpy as np
from keras.models import Sequential
from keras.models import Input
from keras.layers import Dense
from keras.layers import Dropout
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import scipy.stats
import tensorflow as tf
import time
import graphviz
import pydot


# # Load data

# In[207]:


#Import the files
train_df= pd.read_csv("C:/Users/Celia/Desktop/Raph/Scans/merge_table_alltogether_debug_runs.csv", sep=" ",  header = None) 

print("Dataset shape:")
print(train_df.shape)


## Generate labels
ngenes = 300 #Number of genes (columns?)
n_split=5  #Value of K for K-fold cross-validation10

# reshape so each row is one simulation/ flatten the images??
#train_df= np.reshape(train_df, (3942,ngenes))
#train_df= np.reshape(train_df, (,300))



# make label array, first half simple (0), second half complex (1)
all_labels = np.full((4000,1), 0)
all_labels[:2000, 0].fill(1)
all_labels[2000: , 0].fill(0)
all_labels


## shuffles rows of dataset (and labels)
def shuffle_elements(data, labels):
    labels = pd.DataFrame(labels)
    data['lab'] = labels
    
    data = data.sample(frac=1).reset_index(drop=True)
    return data.iloc[:, 0:300], data['lab']


train_df, all_labels = shuffle_elements(train_df, all_labels)


# # Define model

# In[225]:


model = Sequential()
model.add(Dense(100, input_shape=(ngenes,), activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(80, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1, activation='sigmoid'))
model.summary()


# # Define function to reset model (used for running kfold training)

# In[226]:


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


# # Train

# In[227]:



#for i in range(2000):
total_accuracy = 0
total_epochs= 100


#K-fold cross-validation
for train_index,test_index in KFold(n_split).split(train_df):
    
        #print('Train: %s | test: %s' % (train_index, test_index))
        x_train,x_test=train_df.iloc[train_index, :],train_df.iloc[test_index, :]
        y_train,y_test=all_labels[train_index],all_labels[test_index]

        #print(x_train.shape) 
        #print(x_test.shape)
              
       
        #Compile the model/ Train the model
        model.compile(loss='binary_crossentropy',
                      optimizer='SGD',
                      metrics=['accuracy'],)
        
        model = shuffle_weights(model)
        
        history=model.fit(x_train, y_train,
                          epochs=total_epochs,
                          validation_data = (x_test, y_test), 
                          verbose=0)
        
        (loss, accuracy) = model.evaluate(x_test,y_test)
        total_accuracy = total_accuracy + accuracy
        
        #print(history.history)
        
        #Accuracy"
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.ylim([0,1.1])
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.clf()
        
        
            
        # Visualize history
        # Plot history: Loss
        plt.plot(history.history['loss'])
        plt.title('Validation loss history')
        plt.ylabel('Loss value')
        plt.xlabel('No. epoch')
        plt.show()


# In[228]:


#Calculate average accuracy of 1 complete run through the K-fold cross-validation
final_accuracy = total_accuracy/n_split
print('final_accuracy', final_accuracy)
        #scores[i] = final_accuracy
        #Calculate mean and CI from 10 runs through K-fold
        #(mean, lower, upper) = mean_confidence_interval(scores)
        #Append scores from model with current number of nodes
        #data.append([nodes,mean,lower,upper,(upper-lower)])


# In[ ]:




