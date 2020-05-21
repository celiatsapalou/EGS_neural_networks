# -*- coding: utf-8 -*-
"""
Created on Fri Mar 27 14:33:11 2020

@author: Celia
"""

#Loop through each simulation folder
import os
#import collections
#import re
import numpy as np
import pandas as pd
#import csv
#import glob


def readFile(filename, first_col=-1, last_col=-1):
    
    #Read the file from the address
   with open(filename, "rb") as file:  
       data = file.read() # still binary
       
       data = np.frombuffer(data, np.float64)
       if (first_col == -1 & last_col == -1):
           return data
       #if (first_col != -1 & last_col == -1):
           #return data[first_col]
    
       return data[first_col:last_col+1]
   
    
# Extract timepoint index
t=99 
nloci = 300
start = t * nloci
end = start + nloci - 1

#Turn dat file to csv
 #with open(filename, "rb") as dat_file, open(filename, 'w') as csv_file:
     #csv_writer = csv.writer(csv_file)


#with open("filename.dat") as f:
   # with open("filename.csv", "w") as f1:
       # for line in f:
           # f1.write(line)



cwd = os.getcwd()  # Get the current working directory (cwd)
files = os.listdir(cwd)


# This is where the execution code begins:
os.chdir("C:/Users/Celia/Desktop/Raph/Scans")
rootdir = "C:/Users/Celia/Desktop/Raph/Simulations_Interactions"

# Create an empty table (filled with zeros) that will contain those data
data = readFile("C:/Users/Celia/Desktop/Raph/Simulations_Interactions/sim_scaleI_1_1_1_ecosel_1.4_hsymmetry_0_r22/genome_Fst.dat")
#X = np.empty(shape=[3, len(data)])

Y = pd.DataFrame()


# Needs the right dimensions

# Loop through simulation folders
for subdir in os.listdir(rootdir):
    
    # Address of the file
    path = os.path.join(rootdir, subdir, "genome_Fst.dat")
    
     # Read the file
    data = readFile(path, start, end)
    newRow = pd.DataFrame([data])

    
    # Read the file
    #data = readFile(path)
    #newRow = pd.DataFrame([data])
    
    # Append to the table
    Y = Y.append(newRow, ignore_index = True)

    # here i am in the loop
    
print("Imported dataset")
print(Y)

print()
print()
print()
print("-----------------------------------------")
print("-----------the end-------------")
print("-----------------------------------------")

# here i am out of the loop
#Store the scan's content in a temporary variable and 
np.savetxt("C:/Users/Celia/Desktop/Raph/Scans/output_table_I_100_final.csv", Y)