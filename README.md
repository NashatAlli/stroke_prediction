# stroke_prediction
ML-zoomcamp-midterm-project
 ---- 
## Problem description
According to the World Health Organization (WHO), stroke is the second leading cause of death globally, and is responsible for approximately 11% of all deaths.
This data set is used to predict whether a patient is likely to have a stroke based on input parameters from the dataset. The dataset size is 5110 rows, where each row in the data provides relevant information about the patient, and 12 columns, each providing a different feature.

In this project, an ML model predicts the risk likelihood (the possibility of a potential risk occurring, interpreted using qualitative values such as low, medium, or high) of a stroke for a patient based on age, heart disease, hypertension, avg_glucose_level, and smoking status; these five features were found after EDA to be the most features that tell more on the risk of a stroke for patients. With accuracy of 86% that can be further improved by more feature egineering and ML paramter tuning.

(Note: Feature engineering refers to manipulation — addition, deletion, combination, mutation — of your data set to improve machine learning model training, leading to better performance and greater accuracy.)

With an increased synergy between technology and medical diagnosis, caregivers create opportunities for better patient management by systematically mining and archiving the patients’ medical records. Therefore, by deploying this ML model in the deployment process, the hospitals can use this ML service by estimating the likelihood of a stroke for their patients and taking the necessary action.
## Exploratory data analysis (EDA)
### A quick view on the dataset
This is done by first take a view at different random samples from the data. 
Then, take a view at the numerical data by df.describe() to see the count ,the five number summary, the mean and the std(standard deviation).
Finally, view information about the dataset by running the df.info() which prints information about a Dataset including the number of columns, column labels, column data types, memory usage, range index, and the number of cells in each column (non-null values). 

Findings from this quick view: 
- some categorical features(like, hypertension, heart disease and stroke) datatypes are numerical ones, and not object.
- There is non-null values in all of the columns which is okay but must find what is used to fill missing values(for ex, either filled it with zeros or nan)
### EDA on Categorical features
EDA is done to know what categorical features are important in my stroke prediction. 
First, Change the datatypes of hypertension and heart disease from int64 to object.
(hypertension: 0 if the patient doesn't have hypertension, 1 if the patient has hypertension 
heart_disease: 0 if the patient doesn't have any heart diseases, 1 if the patient has a heart disease)

note: (Leaving stroke feature as int64 ,it's 1 if the patient had a stroke or 0 if not, and this will help us in the upcoming steps.)

Secondly, proceed the EDA of the categorical features

- The missing categorical values is filled with "unkown" and this information is available in the kaggle information section about dataset. 
 
note: when saying calculate the mean, it's the mean of stroke column calculated for different groups and globally for all the dataset. 

- Calculate the mean related to different groups, this mean tells us the stroke risk that some group could have in other words, it  tells us the stroke risk likelihood of indiviuals belong to that group could have.
 
- Calculate the overall risk ratio of different groups, which is simply dividing the individual mean of each group by the global stroke mean of the dataset.

- Calculate the mutual information for different groups.





