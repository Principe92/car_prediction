#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  5 18:19:56 2021

@author: Peter
"""

#Script that generates random data for POC
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import string

#Generating random car labels
df1_label=pd.DataFrame(np.arange(4500)).applymap(lambda x: np.random.choice(list(string.ascii_letters)))
#df2_label=pd.DataFrame(np.arange(2250)).applymap(lambda x: np.random.choice(list(string.ascii_letters)))










#Generating a 4500 length data set to simulate Car Part 2 data size it will have a 512 output vector like Resnet18
#using numpy's randint
df1 = pd.DataFrame(np.random.randint(0,100,size=(4500, 512))) #does 512 include label? I don't think it does
#df2 = pd.DataFrame(np.random.randint(0,100,size=(2250, 512))) #how are we going to make sure applicable pairing occurs? (no guarantee there is ever a match, do we make 50% automatically matches for training), we can caluclate how many are matches based on same name before we train





#Performing Dimensonality Reduction with PCA
pca=PCA(n_components=100)

#Df1 PCA
reducedDf1=pd.DataFrame(data=pca.fit_transform(df1))
reducedDf1['label']=df1_label

#Df2 PCA
#reducedDf2=pd.DataFrame(data=pca.fit_transform(df1))
#reducedDf2['label']=df2_label

#Combining the two data sets

#Randomly matching 50% of the total data
mixed_data=pd.DataFrame()
count=0
while count<2250/2:
    index=np.random.randint(0,len(reducedDf1))
    
    first_row=reducedDf1.iloc[index]
    first_label=reducedDf1['label'][index] #getting the first label
    #print(first_label)
    
    reducedDf1.drop(index,inplace=True)
    reducedDf1.reset_index(inplace=True,drop=True) #Dropping the original row so it isn't found multiple times
    
    for row in range(0,len(reducedDf1)):
        if reducedDf1['label'][row]==first_label:
            second_row=reducedDf1.iloc[row]
            #print(second_row)
            reducedDf1.drop(row) #Dropping the original row so it isn't found multiple times
            #Combining the two rows into one and adding the match label
            first_row=first_row.append(second_row)
            first_row['Match']=1
            first_row=first_row.reset_index(drop=True)
            
            mixed_data=pd.concat([mixed_data,first_row],axis=1,ignore_index=True) #adding the original to the new df
            #print(mixed_data)
            count=count+1
            print(count)
            break
        
#Transposing Mixed_Data into the right orientation
mixed_data=mixed_data.T       
mixed_data=mixed_data.rename(columns={100:'label1',201:'label2',202:'Match'})
columns_name=mixed_data.columns

#Matching the other half of the data randomly (how random depends on if the data was shuffled before hand)
df_first_half=reducedDf1[0:1125]
df_first_half.reset_index(inplace=True,drop=True)
df_second_half=reducedDf1[1125:2251]
df_second_half.reset_index(inplace=True,drop=True)
 
mismatched_df=pd.concat([df_first_half,df_second_half],axis=1,join="inner") #combining the unpaired data
mismatched_df['Match']=0
mismatched_df.columns=columns_name

#%%
#Combining the data into a full set again and shuffling it
full_data=pd.concat([mismatched_df,mixed_data])
    
    
    
    
    
    
    
    
    
    
    