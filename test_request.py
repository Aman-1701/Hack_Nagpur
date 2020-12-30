# -*- coding: utf-8 -*-
"""
Created on Wed Dec 30 11:34:16 2020

@author: AMAN VERMA
"""
dictionary = {'name':'Aman Verma', 
              'Age' : '60',
              'Gender': 'Female',
              'Chest Pain': 'Typical Angina',
              'Resting Blood Pressure' : '145',
              'Cholesterol' : '282',
              'Fasting blood Sugar' : '111',
              'Resting Electrocardiographic Result':'0',
              'Maximum heart rate achieved' : '142',
              'Exercise induced Angina':'No',
              'ST depression induced by exercise':'2.3',                         
              'Slope':'Upsloping',       
              'Number of major blood vessels':'1' 
              }
import requests

url = 'https://mod-heart.herokuapp.com/'
r = requests.post(url,json=dictionary)
r.text.strip()
