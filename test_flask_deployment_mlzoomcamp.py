#!/usr/bin/env python
# coding: utf-8
import requests
url= 'http://localhost:7777/predict'

patient={'gender':'Female',
'age':70.0,
'hypertension ':'yes',
'heart_disease':'yes',
'ever_married':'Yes',
'work_type':'Private',
'residence_type':'Urban',
'avg_glucose_level':100.57,
'bmi':25.7,
'smoking_status':'smokes'
}

response = requests.post(url, json=patient).json()
print(response)

