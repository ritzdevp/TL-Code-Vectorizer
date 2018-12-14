# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 00:19:45 2018

@author: Rituraj
"""

#Software Defect Prediction 
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import recall_score
from imblearn.over_sampling import SMOTE



def my_sdp_preprocessor(datafilename_as_csv_inquotes):
    original_data = pd.read_csv(datafilename_as_csv_inquotes)
    original_data.isnull().values.any() #Gives false ie:No null value in dataset
    
    original_X = pd.DataFrame(original_data.drop(['defects'],axis=1))
    original_Y = original_data['defects']
    
    sm = SMOTE(random_state=12, ratio = 1.0)
    x, y = sm.fit_sample(original_X, original_Y)
    y_df = pd.DataFrame(y, columns=['defects'])
    x_df = pd.DataFrame(x, columns=original_X.columns)
    
    combined_training_data = x_df
    combined_training_data['defects'] = y_df
    
    
    original_Y.describe()
    y_df.describe()
    
    import seaborn as sns
    corr = combined_training_data.corr()
    sns.heatmap(corr, xticklabels=corr.columns,yticklabels=corr.columns)
    
    x_train, x_val, y_train, y_val = train_test_split(x_df, y_df, test_size = .1,
                                                              random_state=12)
    original_Y = pd.DataFrame(original_Y)
    
    return original_data, original_X, original_Y,combined_training_data,x_train, x_val, y_train, y_val 


