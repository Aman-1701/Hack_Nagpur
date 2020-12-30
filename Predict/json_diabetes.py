# -*- coding: utf-8 -*-
"""
Created on Fri Dec 25 17:28:09 2020

@author: AMAN VERMA
"""
import json
import joblib
import numpy as np

'''
Code to test stuff
'''
dictionary = {'name':'Aman Verma', 
              'Age' : '31',
              'Gender': 'Male',
              'Pregnacies': '1',
              'Glucose' : '148',
              'Blood Pressure' : '72',
              'Insulin' : '0',
              'Height':'5',
              'Weight' : '65',
              'father':'no',
              'mother':'no',
              'gfather':'no',
              'gmother':'no',
              'mgfather':'no',
              'mgmother' :'no'
              }
json_obj = json.dumps(dictionary,indent=4)

'''
Actual Codes begin
'''
# functioon to preprocesss data
'''
@ input  : dictionary 
@ returns : array of features 
'''
def preprocess(d):
    bmi = (float(d['Weight'])/float(d['Height'])**2) 
    if(d['father']==d['mother']==d['gfather']==d['gmother']==d['mgfather']==d['mgmother']=='yes'):
        pdf = 0.9
    elif(d['father']==d['mother']==d['gfather']==d['gmother']==d['mgfather']=='yes' or
         d['father']==d['mother']==d['gfather']==d['gmother']==d['mgfather']==d['mgmother']=='yes'):
        pdf=0.79
    elif(d['father']==d['mother']==d['gfather']=='yes' or d['father']==d['mother']==d['gmother']=='yes'
         or d['father']==d['mother']==d['mgfather']=='yes' or 
         d['father']==d['mother']==d['mgmother']=='yes'):
       pdf=0.6
    elif(d['father']=='yes' or d['mother']=='yes'):
       pdf=.5
    else: 
       pdf=0.2

    li =np.array([float(d['Pregnacies']), float(d['Glucose']),float(d['Blood Pressure']), float(d['Insulin']),
              bmi, pdf, float(d['Age'] )]).reshape(1,7)
    
    return(li)

# function to predict
'''
input :  array of features
returns : dictionary of form{'result' : '-------'}
'''
def predict(d,li,model):
    loadedmodel= joblib.load(model)
    result = loadedmodel.predict(li)

    if(result==1):
        res = {'Result' : 'Sorry !! {} you are predicted at risk, Must Consult to doctor... Get Well Soon !!'.format(d['name'])}
    else:
        res = {'Result' : 'Woah !! {} you are fine. Have a good day Ahead !!'.format(d['name'])}
    return(res)

def main():
    d=json.loads(json_obj)
    #alll for preprocessing
    li=preprocess(d)
    #calll for prediction
    res=predict(d,li,r'E:/spyder/New Folder/Pima-Indians-Diabetes-kaggle/diabetes.bin')

    print(res['Result'])
    
main()