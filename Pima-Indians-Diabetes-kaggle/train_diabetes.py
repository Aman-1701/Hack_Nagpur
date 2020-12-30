# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:03:56 2020

@author: AMAN VERMA
"""

import pandas as pd
from sklearn import model_selection
from sklearn import ensemble
import joblib

df = pd.read_csv(r"E:/spyder/New folder/Pima-Indians-Diabetes-kaggle/diabetes.csv")
list=['SkinThickness']
df=df.drop(list,axis=1)
df.describe()


x= df.iloc[:,:-1]
y= df.iloc[:,7]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20)

'''
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)
'''
classifier = ensemble.RandomForestClassifier()
classifier.fit(x_train,y_train)

y_pred=classifier.predict(x_test)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

joblib.dump(classifier, 'diabetes.pkl') 
