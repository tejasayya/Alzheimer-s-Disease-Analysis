# Alzheimer's Predictive Analytics
## ITCS 6162 KDD Group 1


[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1G-9fGd6_507aWwMgHHmE1JI2cSEnG-Ht?usp=sharing)

## Team Members
Teja Swaroop Sayya<br>
Venkata Kiran karnam<br>
Ramesh Venkat Reddy Konda<br>
Ruchitha Reddy Katta<br>
Rohith Reddy Alla<br>


## Introduction of Project
The dataset we are working with comes from a detailed study on Alzheimer's Disease in elderly patients. It includes information on people aged 60 to 90 from different backgrounds and education levels. The data covers a wide range of factors such as BMI, smoking habits, alcohol use, physical activity, diet, sleep quality, medical history, blood pressure, cholesterol levels, cognitive test scores (like MMSE), and various symptoms and diagnoses related to Alzheimer's Disease.

Our main goal is to build models that can predict whether someone will be diagnosed with Alzheimer's Disease based on this information. We'll use different types of classification algorithms, like logistic regression, decision trees, random forests, and gradient boosting, to do this. We will also look for the most important factors that influence the risk of getting Alzheimer's. The results from this work could help in detecting Alzheimer's Disease early and improving how it is managed and treated.

### Problem Understanding
This analysis will focus on identifying the characteristics of people who are diagnosed with Alzhiemers disease.

#### Research Question
How do factors like alcohol consumption, age, ethnic background, and medical history affect the risk of developing Alzheimer's disease? Can these variables be used to create a model that identifies individuals at high risk for the condition? What potential benefits could such a model offer for early intervention and care?

https://www.washingtonpost.com/wellness/2024/03/27/dementia-aging-risk-brain-diabetes-pollution-alcohol/

What characteristics are most common in people who become diagnosed with Alzheimer's disease?
Which of the characteristics can be identified prior to the onset of the disease?
https://www.nia.nih.gov/health/alzheimers-and-dementia/alzheimers-disease-fact-sheet https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8927715/

### Data Source and Description:

The Alzheimer’s Disease Dataset was selected:
https://www.kaggle.com/datasets/rabieelkharoua/alzheimers-disease-dataset?select=alzheimers_disease_data.csv

The dataset contains information on 2,150 patients who were checked for Alzheimer's Disease, noting whether they were diagnosed with the illness. Some data, like gender and ethnicity, are categories, while other information includes numbers that rate conditions like sleep quality or measure things like cholesterol levels. These numbers change over time as patients age. Some parts of the dataset, such as the doctor's name, have been removed and don't provide useful information.

This dataset helps us understand how different factors are related to the risk of Alzheimer's Disease. By analyzing it, we can develop models to predict the likelihood of a diagnosis, which could lead to better early detection and treatment strategies.


### Data Understanding and EDA:
Certainly! It seems that lifestyle factors and cognitive features are more strongly correlated with the prediction of Alzheimer's disease. This suggests that how a person lives and their cognitive abilities may provide important clues about their risk of developing Alzheimer's.


##### Exploratory Data Analysis (EDA) reveals key insights:
- Age: Most patients are between 70-80 years old.
- Gender: Comparison of Alzheimer's diagnoses between males and females.
- Ethnicity and Education: Analysis of diagnosis rates across ethnic groups and education levels.
- Lifestyle Factors: Box plots showing BMI, alcohol consumption, physical activity, diet, and sleep quality differences by diagnosis status.
- Medical History: Prevalence of medical conditions in diagnosed vs. non-diagnosed patients.
- Clinical Measurements: Blood pressure and cholesterol level distributions.
- Cognitive Assessments : Differences in MMSE, functional assessment, and ADL scores.
- Symptoms: Frequency of confusion, disorientation, and other symptoms in diagnosed patients.

These insights inform the selection of features and algorithms for predictive modeling. 




### Data Preparation
The categorical columns of the Alzheimer’s Disease Dataset could be one-hot encoded during data preprocessing. Any of the numeric columns could be standardized such that they reflect a normal distribution if they are found to be normally distributed while others could be linearly scaled to reduce the effect of bias in any machine learning models for the dataset. The sanitized columns in the dataset could also be dropped since they provide no useful information.

##### Data  
- We are removing the columns PatientId and DoctorInCharge
- Data is highly weighted to non-smokers. Looking to see if there is a correlation between smoking data to non smokers.
- Family history. Also highly weighted.  Look for a correlation.
- The medical history tends to focus on the negative aspects of each condition. It might be helpful to look for any patterns or connections between them. Additionally, memory complaints and behavioral problems are very subjective and might be better to leave out.


