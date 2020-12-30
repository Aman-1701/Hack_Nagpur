# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 23:53:44 2020

@author: AMAN VERMA
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler 
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

df = pd.read_csv(r'E:/spyder/New folder/Classification-of-heart-disease/heart.csv')

X = df.drop(['target','thal'], axis=1)
y = df.target

# Encode the categorial feature
X_encoded = pd.get_dummies(X, columns=['sex', 'cp','fbs','restecg','exang','slope','ca'])
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state = 1)


classifier = DecisionTreeClassifier()
classifier.fit(X_train,y_train )
y_pred_classifier = classifier.predict(X_test)
classifier_cm = confusion_matrix(y_test, y_pred_classifier)
print(classifier_cm)

joblib.dump(classifier, 'heart.pkl') 