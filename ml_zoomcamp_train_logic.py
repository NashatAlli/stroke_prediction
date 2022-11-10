# Import dependencies
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression

import pickle

output_file = f'modellog6.bin'

# ## Reading dataset

df= pd.read_csv('healthcare-dataset-stroke-data.csv')


# # EDA 

# ## EDA for categorical features


heart_disval={
    0:'no',
    1:'yes'
}

hypertension_val={
    0:"no",
    1:'yes'
}


df.heart_disease= df.heart_disease.map(heart_disval)
df.hypertension=df.hypertension.map(hypertension_val)

df.columns=df.columns.str.lower()

categorical=['gender','hypertension','heart_disease','ever_married','work_type','residence_type','smoking_status']

df_used=df.copy()

# ### Chosen features
numerical_chosen=['age','avg_glucose_level']
categorical_chosen=['hypertension','heart_disease','smoking_status']
del df_used['id']

# # Training,validation and testing steps

# ### spliting our dataset with train_test_split

df_full_train, df_test= train_test_split(df_used, test_size= 0.2, random_state= 1 )
df_train, df_val= train_test_split(df_full_train, test_size= 0.25, random_state=1)

df_train= df_train.reset_index(drop= True)
df_val= df_val.reset_index(drop=True)
df_test=df_test.reset_index(drop=True)

y_train=df_train.stroke.values
y_val=df_val.stroke.values
y_test=df_test.stroke.values

del df_train['stroke']
del df_test['stroke']
del df_val['stroke']

# ### Logistic regression

def train_logistic(df):
    dicts_train = df[categorical_chosen+numerical_chosen].to_dict(orient='records')
    dv= DictVectorizer(sparse=False)
    X_train= dv.fit_transform(dicts_train)
    y_train= df.stroke.values
    model= LogisticRegression(solver='liblinear', C=1.0, max_iter= 1000)
    model.fit(X_train, y_train)
    
    return X_train, y_train,dv, model

def predicition_logistic(df,dv, model):
    dicts_val = df[categorical_chosen+numerical_chosen].to_dict(orient='records')
    X_val= dv.transform(dicts_val)
    y_predict= model.predict_proba(X_val)
    
    return y_predict


# ### spliting and validation by cross-validation

print('\ncross-validation starts\n')

kfold = KFold(n_splits=5, shuffle=True, random_state=1)

auc_scores= []

for train_idx, val_idx in kfold.split(df_full_train):
    df_train2 = df_full_train.iloc[train_idx]
    df_val2= df_full_train.iloc[val_idx]
    X_train2, y_train2, dv, model_log2= train_logistic(df_train2)
    y_val2= df_val2.stroke.values
    
    y_predict6= predicition_logistic(df_val2, dv, model_log2)
    y_predict6 = y_predict6[:,1] 
    
    auc = roc_auc_score(y_val2,y_predict6)
    auc_scores.append(auc)
    
    print("mean: %.3f & std : %.3f " %(np.mean(auc_scores),np.std(auc_scores)))


# Training the final model

print('\nTraining the final model \n')
X_train5, y_train5, dv5, model_log5 = train_logistic(df_full_train)
y_predict9 = predicition_logistic(df_test, dv5, model_log5)

y_predict9 = y_predict9[:, 1]

auc5=roc_auc_score(y_test, y_predict9)

print(f'auc={auc5}')

# Save the Model
f_out=open(output_file,'wb')
pickle.dump((dv5, model_log5),f_out)
f_out.close()

print(f'\nThe model is saved to {output_file}\n')
