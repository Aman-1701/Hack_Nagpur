# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:44:20 2020

@author: AMAN VERMA
"""
import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

df = pd.read_csv(r'E:/spyder/New folder/Prediction-on-Indian-Liver-Patient-Data/liver_patient_data.csv')
df.isnull().values.any()
df["Albumin_and_Globulin_Ratio"].fillna(df['Albumin_and_Globulin_Ratio'].mean(), inplace = True)
df.isnull().values.any()

# encode the categorial features
def partition(x):
    if x =='Male':
        return 0
    return 1

df['Gender'] = df['Gender'].map(partition)

X = df.drop('Dataset',axis=1)
y = df['Dataset']

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=123)

model = ensemble.RandomForestClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print('Accuracy : {}'.format(accuracy_score(y_test, y_pred)))

joblib.dump(model,"liver.pkl")
