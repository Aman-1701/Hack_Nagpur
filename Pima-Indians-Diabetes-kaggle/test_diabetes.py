# -*- coding: utf-8 -*-
"""
Created on Wed Dec 16 19:16:12 2020

@author: AMAN VERMA
"""
import joblib
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

loadedmodel= joblib.load(r'E:/spyder/New folder/Pima-Indians-Diabetes-kaggle/diabetes.pkl')
df = pd.read_csv(r"E:/spyder/New folder/Pima-Indians-Diabetes-kaggle/diabetes.csv")
list=['SkinThickness']
df=df.drop(list,axis=1)
df.describe()

xp=np.array([6,148,72,0,33.6,0.627,50]).reshape(1,7)


x= df.iloc[:,:-1]
y= df.iloc[:,7]
x_train,x_test,y_train,y_test=model_selection.train_test_split(x,y,test_size=0.20)

sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

classifier=LogisticRegression(random_state=0)
classifier.fit(x_train,y_train)

y_pred=classifier.predict(xp)

result = loadedmodel.predict(xp)
