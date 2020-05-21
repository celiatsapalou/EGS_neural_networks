# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:52:08 2020

@author: Celia
"""

import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from matplotlib import pyplot as plt
from sklearn.model_selection import KFold
import pandas as pd
import scipy.stats
import tensorflow as tf
#import os
import time
#import mnist

#start measuring time
t0 = time.time()

#os.environ["CUDA_VISIBLE_DEVICES"]="-1"   

#Import the files
train_df= pd.read_csv("C:/Users/Celia/Desktop/Raph/Scans/merge_output_table_all_finalspace.csv", header = None) 
#train_df=tf.data.experimental.make_csv_dataset("C:/Users/Celia/Desktop/Raph/Scans/merge_output_table_100_final.csv", batch_size=20, header=False)
print(train_df.shape)
#pd.isnan(train_df)
# 0000111101001010 10101010010101010 10101010101010101 ... -> one genome scan

#read in data using pandas
train_df = train_df

#check data has been read in properly
print(train_df)


ngenes = 300 #Number of genes (columns?)
n_split=10  #Value of K for K-fold cross-validation10

# reshape so each row is one simulation/ flatten the images??
#train_df= np.reshape(train_df, (20,ngenes))


# make label array, first half simple (0), second half complex (1)
all_labels = np.full((2000,2), 0)
all_labels[:1000, 0].fill(1)
all_labels[1000: , 1].fill(1)


# prepare data for normalization
#values = series.values
#values = values.reshape((len(values), 1))
# train the normalization
#scaler = MinMaxScaler(feature_range=(0, 1))


# 1 0
# 1 0
# 1 0
# ...
# 0 1
# 0 1
# 0 1

#Returns mean and confidence interval from data
#def mean_confidence_interval(data, confidence=0.95):
   #a = 1.0 * np.array(data)
   # n = len(a)
   # m, se = np.mean(a), scipy.stats.sem(a)
   # h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
   # return m, m-h, m+h

#x_train=tf.keras.utils.normalize(x_train, axis=1)
#x_test=tf.keras.utils.normalize(x_test, axis=1)
#model=tf.keras.models.Sequential()
#model.add(tf.keras.layers.Flatten()) 

#Start training and testing

scores = np.empty(2000)


#add model layers
#model = Sequential()
##model.add(Dense(nodes, input_dim=300, activation='sigmoid'))
#model.add(Dense(15))
#model.add(Dense(2, activation='softmax'))
#model.summary()

#for i in range(2000):
total_accuracy = 0
total_epochs= 20


#K-fold cross-validation
for train_index,test_index in KFold(n_split).split(train_df):
        print('Train: %s | test: %s' % (train_index, test_index))
        x_train,x_test=train_df.iloc[train_index, :],train_df.iloc[test_index, :]
        y_train,y_test=all_labels[train_index,:],all_labels[test_index, :]
        #print(x_train.shape) 
        #print(x_test.shape)
      
        model = Sequential()
        model.add(Dense(40, input_dim=300, activation='sigmoid'))
        model.add(Dense(30, activation='sigmoid'))
        model.add(Dense(30, activation='sigmoid'))
        model.add(Dense(2, activation='softmax'))
        model.summary()
       
        #Compile the model/ Train the model
        model.compile(loss='binary_crossentropy',
                      optimizer='nadam',
                      metrics=['binary_accuracy'],)
        
        history=model.fit(x_train, y_train,epochs=total_epochs,validation_data = (x_test, y_test))
        (loss, accuracy) = model.evaluate(x_test,y_test)
        total_accuracy = total_accuracy + accuracy
        #print(history.history)
        
        #Accuracy"
        plt.plot(history.history['binary_accuracy'])
        plt.plot(history.history['val_binary_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'validation'], loc='upper right')
        plt.show()
        plt.clf()
            
#Calculate average accuracy of 1 complete run through the K-fold cross-validation
final_accuracy = total_accuracy/n_split
print('final_accuracy', final_accuracy)
        #scores[i] = final_accuracy
        #Calculate mean and CI from 10 runs through K-fold
        #(mean, lower, upper) = mean_confidence_interval(scores)
        #Append scores from model with current number of nodes
        #data.append([nodes,mean,lower,upper,(upper-lower)])

print(train_df.shape)
#Export data to csv
#df = pd.DataFrame(, columns = ['Nodes', 'Accuracy', 'CI min', 'CI max', 'CI size']) 
#df.to_csv('Model 1.csv')

#Print runtime
#print(time.time() - t0, "seconds wall time")

#data.to_csv("C:/Users/Celia/Desktop/Raph/Scans/output_table_final.csv", )