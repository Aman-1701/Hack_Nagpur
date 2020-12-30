# -*- coding: utf-8 -*-
"""
Created on Thu Dec 17 17:52:22 2020

@author: AMAN VERMA
"""

import numpy as np
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

loadedmodel=joblib.load('liver.pkl')
xp=np.array([38,1,0.8,0.2,185,25,21,7.0,3.0,0.7]).reshape(-1,10)

result = loadedmodel.predict(xp)