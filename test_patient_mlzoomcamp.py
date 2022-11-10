#!/usr/bin/env python
# coding: utf-8
import pickle
model_file = 'modellog5.bin'

with open(model_file,'rb') as f_in:
    dv,model= pickle.load(f_in)

patient={'gender':'Female',
'age':47.0,
'hypertension ':'yes',
'heart_disease':'no',
'ever_married':'Yes',
'work_type':'Private',
'residence_type':'Urban',
'avg_glucose_level':100.57,
'bmi':25.7,
'smoking_status':'smokes'
}

patient_features=dv.transform([patient])

y_pred= model.predict_proba(patient_features)[:,1]

print('patient ' , patient,'\n')

print('stroke_prediction:',float(y_pred),'\n')

print('High stroke risk likelihood')if 0.5<=y_pred else print('Medium stroke risk likelihood') if 0.5>y_pred>=0.4 else print('Low stroke risk likelihood')
