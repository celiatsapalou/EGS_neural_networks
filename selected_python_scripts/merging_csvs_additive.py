# -*- coding: utf-8 -*-
"""
Created on Mon Mar 30 13:13:25 2020

@author: Celia
"""
import os
import pandas as pd
#import numpy as np

os.chdir("C:/Users/Celia/Desktop/Raph/Scans")

a = pd.read_csv("C:/Users/Celia/Desktop/Raph/Scans/merge_table_A_debug_alltogether_runs.csv", sep = " ", header = None)
b = pd.read_csv("C:/Users/Celia/Desktop/Raph/Scans/output_table_A_debug_to_add.csv", sep = " ", header = None)
print(a)
print(b)



data = a.append(b)
print(data)

# append complex data to simple
#all_data = np.append(a,b)
data.to_csv("C:/Users/Celia/Desktop/Raph/Scans/merge_table_A_debug_alltogether_runs.csv", sep = " ", header = False, index = False)



