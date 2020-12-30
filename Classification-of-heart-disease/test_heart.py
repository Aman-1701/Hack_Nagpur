# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 00:02:47 2020

@author: AMAN VERMA
"""

import joblib
import pandas as pd
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
import numpy as np

loadedmodel= joblib.load(r'E:/spyder/New folder/Classification-of-heart-disease/heart.bin')
xp=np.array([52,118,186,190,0.0,0,1,0,0,0,1,1,0,1,0,0,1,0,0,1,0,1,0,0,0,0]).reshape(1,26)


result = loadedmodel.predict(xp)

	

