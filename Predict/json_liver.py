# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 18:27:29 2020

@author: AMAN VERMA
"""

import json
import joblib
import numpy as np
'''
Code to test stuff
'''
dictionary = {'name':'Aman Verma', 
              'Age' : '17',
              'Gender': 'Male',
              'Total Bilirubin': '0.9',
              'Direct Bilirubin' : '0.3',
              'Alkaline Phosphate' : '202',
              'Alamine Aminotransferase' : '22',
              'Aspartate Aminotransferase':'19',
              'Total Protein' : '7.4',
              'Albumin':'4.1',
              'Albumin : Globulin Ratio':'1.2'              
              }
'''male=0 female=1'''

json_obj = json.dumps(dictionary,indent=4)
d = json.loads(json_obj)
'''
Actual Codes begin
'''
# functioon to preprocesss data
'''
@ input  : dictionary 
@ returns : array of features 
'''
def preprocess(d):
    gender= 0 if d['Gender']=='Male' else 1
    li =np.array([float(d['Age']), gender, float(d['Total Bilirubin']),float(d['Direct Bilirubin']), 
              float(d['Alkaline Phosphate']),float(d['Alamine Aminotransferase']), 
              float(d['Aspartate Aminotransferase']),float(d['Total Protein']),
              float(d['Albumin']),float(d['Albumin : Globulin Ratio'])]).reshape(1,10)
    return(li)

# function to predict
'''
input :  array of features
returns : dictionary of form{'result' : '-------'}
'''
def predict(li):
    loadedmodel=joblib.load(r'E:/spyder/New Folder/Prediction-on-Indian-Liver-Patient-Data/liver.pkl')
    result =  loadedmodel.predict(li)
    if(result==1):
        res = {'Result' : 'Sorry !! {} you are predicted at risk, Must Consult to doctor... Get Well Soon !!'.format(d['name'])}
    else:
        res = {'Result' : 'Woah !! {} are fine. Have a good day Ahead !!'.format(d['name'])}
    return(res)

d = json.loads(json_obj)
li= preprocess(d)
res=predict(li)
print(res['Result'])