# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 03:14:37 2018

@author: Rituraj
"""

import pandas as pd
import preprocessingfile as preprocess
import models

data = 'pc2.csv'
original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val = preprocess.my_sdp_preprocessor(data)
all_data = [original_data, original_X, original_Y,combined_training_data,x_train1,x_train2,x_train,x_test,x_val,y_train1,y_train2,y_train,y_test,y_val]

nn_clf = models.NN(*all_data)
cnn_clf = models.cnn(*all_data)
svm_clf = models.svm(*all_data)
rf_clf = models.random_forest(*all_data)