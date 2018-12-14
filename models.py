# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 01:13:37 2018

@author: Rituraj
"""
import pandas as pd
import preprocessingfile as preprocess
original_data, original_X, original_Y,combined_training_data,x_train, x_val, y_train, y_val = preprocess.my_sdp_preprocessor('nasadataKC1.csv')

def NN_3_layers():   
    # Importing the Keras libraries and packages
    import keras
    from keras.models import Sequential
    from keras.layers import Dense
    
    # Initialising the ANN
    classifier = Sequential()
    
    # Adding the input layer and the first hidden layer
    classifier.add(Dense(output_dim = 15, init = 'uniform', activation = 'relu', input_dim = 22))
    # Adding the second hidden layer
    classifier.add(Dense(output_dim = 8, init = 'uniform', activation = 'relu'))
    classifier.add(Dense(output_dim = 5, init = 'uniform', activation = 'relu'))
    # Adding the output layer
    classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))
    # Compiling the ANN
    
    classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
    # Fitting the ANN to the Training set
    classifier.fit(x_train, y_train, batch_size = 10, nb_epoch = 100)
    
    #Making the predictions and evaluating the model
    # Predicting the Test set results
    y_pred = classifier.predict(x_val)
    y_pred = (y_pred > 0.5)
    y_pred = pd.DataFrame(y_pred, columns=['defects'])
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_val, y_pred)
    
    return classifier

def catboost_1():
    from catboost import CatBoostClassifier
    clf = CatBoostClassifier(iterations=100, depth=10, learning_rate=0.001)
    clf.fit(x_train, y_train, plot=True)
    y_pred = clf.predict(x_val)
    y_pred = pd.DataFrame(y_pred, columns=['defects']).astype('bool')
    # Making the Confusion Matrix
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(y_val, y_pred)
    from sklearn.metrics import accuracy_score
    accuracy_score(y_val, y_pred)
    
    return clf
    
    
NN_model = NN_3_layers()
catboost_model = catboost_1()

y_pred1 = NN_model.predict(x_val)
y_pred1 = (y_pred1 > 0.5)
y_pred1 = pd.DataFrame(y_pred1, columns=['defects'])
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_val, y_pred1)
from sklearn.metrics import accuracy_score
print('acc:',accuracy_score(y_val, y_pred1))
print(cm)

y_pred2 = catboost_model.predict(x_val)
y_pred2 = pd.DataFrame(y_pred2, columns=['defects'])
# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm2 = confusion_matrix(y_val, y_pred2)
from sklearn.metrics import accuracy_score
print('acc:',accuracy_score(y_val, y_pred2))
print(cm2)

