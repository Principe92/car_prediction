#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat May  8 15:10:36 2021

@author: Peter
"""
#Script that generates random data for POC
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
import string
from itertools import combinations

#Generating random car labels
df1_label=pd.DataFrame(np.arange(500)).applymap(lambda x: np.random.choice(list(string.ascii_letters)))



#Generating a 4500 length data set to simulate Car Part 2 data size it will have a 512 output vector like Resnet18
#using numpy's randint
df1 = pd.DataFrame(np.random.randint(0,100,size=(500, 512))) #does 512 include label? I don't think it does

#Performing Dimensonality Reduction with PCA
pca=PCA(n_components=100)

#Df1 PCA
data=pd.DataFrame(data=pca.fit_transform(df1))
data['label']=df1_label
data2=data.copy()

#%%
mixed_data=pd.DataFrame()
count=0
stop=0
while stop==0:
    label=data['label'][count]
    index_list=data.index[data['label']==label].tolist()
    possible_combos=list(combinations(index_list,2))
    if len(possible_combos)!=0:
        for i in range(0,len(possible_combos)):
            leftside=data.iloc[possible_combos[i][0]] #Getting values at index (left side of pair)
            rightside=data.iloc[possible_combos[i][1]] #Getting values at index (right side of pair)
            matched_pair=pd.concat([leftside,rightside],axis=0,ignore_index=True)
            mixed_data=pd.concat([mixed_data,matched_pair],axis=1,ignore_index=True)
            #print(i)
        list_to_delete=set([i[0] for i in possible_combos])
        data.drop(list_to_delete,inplace=True)  #Getting rid of all rows with those labels
        data.reset_index(inplace=True,drop=True) #resetting index
        
    else:
        data.drop(index_list,inplace=True)  #Getting rid of all rows with those labels
        data.reset_index(inplace=True,drop=True) #resetting index
        
        
    print(label)
    print(len(data))
    if len(data)==0:
        stop=1

#%%
#Transposing Mixed_Data into the right orientation
mixed_data=mixed_data.T
mixed_data['Match']=1
mixed_data=mixed_data.rename(columns={100:'label1',201:'label2',202:'Match'})
columns_name=mixed_data.columns    

#%%
#Randomly mixing data for the rest
num_non_match=5000
non_match_df=pd.DataFrame()
while len(non_match_df.columns)<num_non_match:
    index1=np.random.randint(0,len(data2))
    index2=np.random.randint(0,len(data2))
    leftside=data2.iloc[index1] #Getting values at index (left side of pair)
    rightside=data2.iloc[index2] #Getting values at index (right side of pair)
    mismatched_pair=pd.concat([leftside,rightside],axis=0,ignore_index=True)
    non_match_df=pd.concat([non_match_df,mismatched_pair],axis=1,ignore_index=True)
    print(len(non_match_df.columns))

#%%
#Transposing non_match_df into the right orientation
non_match_df=non_match_df.T
non_match_df['Match']=0
non_match_df=non_match_df.rename(columns={100:'label1',201:'label2',202:'Match'})
columns_name=non_match_df.columns     

#drop where matches accidently exist
non_match_df=non_match_df[non_match_df['label1']!=non_match_df['label2']]
    
#%%
#Combining the two data sets
full_data=pd.concat([non_match_df,mixed_data]).sample(frac=1)

