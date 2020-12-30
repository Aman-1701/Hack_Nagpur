
import json
import joblib
import numpy as np


def preprocess(d):
    li=[float(d['Age'])]
    li=li+[float(d['Resting Blood Pressure'])]+[(float(d['Cholesterol']))]+[(float(d['Maximum heart rate achieved']))]+[float(d['ST depression induced by exercise'])]

    gen =[1,0] if d['Gender']=='Male' else [0,1]

    if d['Chest Pain']== 'Asymptomatic':
        cp=[1,0,0,0]
    elif d['Chest Pain']== 'Atypical Angina':
        cp=[0,0,0,1]
    elif d['Chest Pain']== 'Non-Anginal Pain':
        cp=[0,0,1,0]
    else:
        cp=[0,1,0,0]
    
    if float(d['Fasting blood Sugar'])> 120:
        fps = [0,1]
    else:
        fps = [1,0]
     
    if d['Resting Electrocardiographic Result'] == 'Normal':
        restcg=[0,1,0]
    elif d['Resting Electrocardiographic Result'] == 'ST-T wave with abnormality':
        restcg=[1,0,0]
    else:
        restcg=[0,0,1]
    
    if d['Exercise induced Angina'] == 'No':
        exang=[1,0]
    else:
        exang=[0,1]

    if d['Slope']=='Upsloping':
        slp=[0,0,1]
    elif d['Slope']=='Flat':
        slp=[0,1,0]
    else:
        slp=[1,0,0]

    if d['Number of major blood vessels'] == '0':
        ca = [1,0,0,0,0]
    elif d['Number of major blood vessels'] == '1':
        ca = [0,1,0,0,0]
    elif d['Number of major blood vessels'] == '2':
        ca = [1,0,0,0,0]
    elif d['Number of major blood vessels'] == '3':
        ca = [0,0,0,1,0]
    
    li=li+gen+cp+fps+restcg+exang+slp+ca
    li=np.array(li).reshape(1,26)
    
    return (li)
    

def predict_result(d,model):
    li = preprocess(d)
    result = model.predict(li)
    
    if(result==1):
        res = {'Result' : 'Sorry !! {} is predicted at risk, Must Consult to doctor... Get Well Soon !!'.format(d['name'])}
    else:
        res = {'Result' : 'Woah !! {} is fine. Have a good day Ahead !!'.format(d['name'])}
    return(res['Result'])
