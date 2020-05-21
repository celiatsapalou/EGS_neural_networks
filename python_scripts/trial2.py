# -*- coding: utf-8 -*-
"""
Created on Mon Mar 16 11:51:41 2020

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

#call this function only with filename so that it returns everything in the file
## i.e. readFile(filename) gives all 30.000 values

# call the function with path and 1 number to return the specified column
## i.e. readFile(filename, 100) gives value in the 100th column only

# call the function with path and 2 numbers to return the specified columns in that range
## i.e. readFile(filename, 100, 300) gives value in the range 100 and 300 
def readFile(filename, first_col = - 1, last_col = -1):
    
    # Read the file from the address
    with open(filename, "rb") as file:  
        data = file.read() # still binary
            
    # Converting binary to numbers    
    data = np.frombuffer(data, np.float64)
    if (first_col == -1 & last_col == -1):
        return data
    if (first_col != -1 & last_col == -1):
        return data[first_col]
    
    return data[first_col:last_col]


#Turn dat file to csv
 #with open(filename, "rb") as dat_file, open(filename, 'w') as csv_file:
     #csv_writer = csv.writer(csv_file)


#with open("filename.dat") as f:
   # with open("filename.csv", "w") as f1:
       # for line in f:
           # f1.write(line)



 #for line in dat_file:
       # row = [field.strip() for field in line.split('|')]
        #if len(row) == 6 and row[3] and row[4]:
            #csv_writer.writerow(row)



# This is where the execution code begins:
os.chdir("C:/Users/Celia/Desktop/Raph/Scans")
rootdir = 'C:/Users/Celia/Desktop/Raph/simulations_A'

# Create an empty table (filled with zeros) that will contain those data
data = readFile("C:/Users/Celia/Desktop/Raph/simulations_A/sim_scaleA_1_1_1_ecosel_1.4_hsymmetry_0_r1/genome_Fst.dat")
#X = np.empty(shape=[3, len(data)])

X = pd.DataFrame()


# Needs the right dimensions

# Loop through simulation folders
for subdir in os.listdir(rootdir):
    
    # Address of the file
    path = os.path.join(rootdir, subdir, "genome_Fst.dat")
    
    # Read the file
    data = readFile(path, 0, 100)
    newRow = pd.DataFrame([data])

    
    # Append to the table
    
    # iloc[row, colunmn] gives data in specified row and column
    # you can specify a range with the column : for example 100:1002
    # get range of rows 100 - 101  ---> newRow.iloc[:,100:102]
    # get range of rows 100 - 201  ---> newRow.iloc[:,100:202]

    X = X.append(newRow, ignore_index = True)

    # here i am in the loop
    
print("Imported dataset:")
print(X)

print()
print()
print()



print("-----------------------------------------")
print("------------Villos----------")
print("-----------------------------------------")

# here i am out of the loop
#Store the scan's content in a temporary variable and 
np.savetxt("C:/Users/Celia/Desktop/Raph/Scans/output_table_A_100.csv", X)
    
    
    
 

    
#Append info to a table
    
#import files
    

    
