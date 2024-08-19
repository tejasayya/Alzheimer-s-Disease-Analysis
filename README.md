# Alzheimer's Predictive Analysis

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/163DsE4Z2fajrk0MHkBHIcnQXgq1LJNS0?usp=sharing)

or

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/gist/tejasayya/c4f1994d4733858a6138ba5296ebee11/alzheimer-s-predictive-analysis.ipynb)


## Project Introduction 
The dataset we are working with comes from a detailed study on Alzheimer's Disease in elderly patients. It includes information on people aged 60 to 90 from different backgrounds and education levels. The data covers a wide range of factors such as BMI, smoking habits, alcohol use, physical activity, diet, sleep quality, medical history, blood pressure, cholesterol levels, cognitive test scores (like MMSE), and various symptoms and diagnoses related to Alzheimer's Disease.

Main goal is to build models that can predict whether someone will be diagnosed with Alzheimer's Disease based on this information. We'll use different types of classification algorithms, like logistic regression, decision trees, random forests, and gradient boosting, to do this. We will also look for the most important factors influencing the risk of Alzheimer's. The results from this work could help detect Alzheimer's disease early and improve how it is managed and treated.

### Problem Understanding
This analysis will focus on identifying the characteristics of people who are diagnosed with Alzheimer's disease.

### Research Question
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
We identified different attributes to gain an understanding of the dataset. 
The lifestyle factors and cognitive features are more strongly correlated with the prediction of Alzheimer's disease.
This suggests that how a person lives and their cognitive abilities may provide important clues about their risk of developing Alzheimer's.

##### Exploratory Data Analysis (EDA) reveals key insights:
- Age: Most patients are in this age Group [61, 68, 75, 82, 89].
![age](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/age.png)


As Data is Huge took only the age bracket between 70 to 73. **outliers are high**.

![agee](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/symptom_counts.png)


- Gender: Comparison of Alzheimer's diagnoses between males and females.
![gender](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/gender.png)
  
- Ethnicity and Education: Analysis of diagnosis rates across ethnic groups and education levels.
- 0: Caucasian
- 1: African American
- 2: Asian
- 3: Other
  
![ethnicity](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/enthnicity.png)


EducationLevel: The education level of the patients, coded as follows:
- 0: None
- 1: High School
- 2: Bachelor's
- 3: Higher
  
![Education](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/education.png)



- Lifestyle Factors: Box plots showing BMI, alcohol consumption, physical activity, diet, and sleep quality differences by diagnosis status.


![lifestyle](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/smoking.png)

![lifestyle](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/AlchoholConsumption.png)

![lifestyle](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/Physical%20Activity.png)

![lifestyle](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/BMI.png)

![lifestyle](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/SleppQuality.png)


- Medical History: Prevalence of medical conditions in diagnosed vs. non-diagnosed patients.


![Behavioural Problems](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/Diagnosis.png)

![memory complaints](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/memorycomplaints.png)

- Clinical Measurements: Blood pressure and cholesterol level distributions.
- Cognitive Assessments: Differences in MMSE, functional assessment, and ADL scores.
- Symptoms: Frequency of confusion, disorientation, and other symptoms in diagnosed patients.



These insights inform the selection of features and algorithms for predictive modelling. 

![insights](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/insights.png)




# Data Preparation
# Data Pre-Processing
We import libraries like numpy, pandas, seaborn, and matplotlib for data manipulation and visualization.
The csv data is loaded into into a pandas DataFrame using the pd.read_csv function.

- Data Inspection:
  - Used df.info() to display basic information about the dataset, including:
    - Number of non-null entries
    - Data types of the columns
    - Memory usage
- Handling Null Values:
  - Checked for null values in the dataset
  - Dropped the small number of null values found
- Data Cleaning:
  - there are outliers.
  - We are removing the columns PatientId and DoctorInCharge
- Data Preprocessing:
  - Used LabelEncoder() to encode categorical variables
  - Applied SMOTE (Synthetic Minority Over-sampling Technique) to balance the data
These initial steps ensure that the data is ready for further analysis, manipulation, and modeling.


##### Data  
- Data is highly weighted to non-smokers. Looking to see if there is a correlation between smoking data to non smokers.
- Family history. Also highly weighted.  Look for a correlation.
- The medical history tends to focus on the negative aspects of each condition. It might be helpful to look for any patterns or connections between them. Additionally, memory complaints and behavioral problems are very subjective and might be better to leave out.


The categorical columns of the Alzheimer’s Disease Dataset could be one-hot encoded during data preprocessing. Any of the numeric columns could be standardized such that they reflect a normal distribution if they are found to be normally distributed while others could be linearly scaled to reduce the effect of bias in any machine learning models for the dataset. The sanitized columns in the dataset could also be dropped since they provide no useful information.




# Modelling

Initial Models
  - Trained and evaluated the following models using all variables except the target variable ('Performance Impact') as predictors:
    - Gradient Boosting Classifier: 95.356 accuracy
    - Random Forest: 87.61 accuracy
    - Logistic Regression: 80.18 accuracy
    - K-Nearest Neighbors: 57.89 accuracy

Pycaret Exploration
  - Implemented and explored Pycaret for automated machine learning

Feature Selection
  - Used PyCaret to choose the best 10 features and re-trained the models:
    - Extreme Gradient Boosting: 0.96 accuracy
    - Logistic Regression: 0.85 accuracy
    - Decision Tree: 0.94 accuracy
    - Random Forest: 0.95 accuracy
    - K-Nearest Neighbors: 0.90 accuracy


Narrowed down the best features to:
    - 'FunctionalAssessment '
    - 'ADL'
    - 'MMSE'
    - 'MemoryComplaints'
    - 'BehavioralProblems'
    - 'CholesterolTotal'
    - 'CholesterolTriglycerides'
    - 'BMI'
    - 'SleepQuality'
    - 'Age'


Hyperparameter Tuning
  - Used GridSearchCV() to tune hyperparameters for each model using all variables except the target variable:
    -  Resulted in lower accuracy scores
   

Used GridSearchCV() to tune hyperparameters for each model using the narrowed-down predictors list:
  - Logistic Regression: 0.23 accuracy
  - Decision Tree: 0.65 accuracy
  - Random Forest: 0.71 accuracy
  - K-Nearest Neighbors: 0.63 accuracy



# Evaluation
 
  Evaluation Metrics
- Accuracy: measures the overall correctness of the classification model
  - Proportion of total correct predictions to the total number of predictions
- Recall: measures how well the model identifies students correctly in each category
  - For each category, recall = (number of correct predictions) / (actual number of students in that category)
- Precision: measures the accuracy of the model's predictions for each category
  - For each category, precision = (number of correct predictions) / (total number of predictions in that category)
- F1-Score: combines precision and recall into a single metric for each category
  - Useful when there is an imbalance in the dataset
- Support: indicates the number of students in each performance impact category in the dataset
  - Provides context for evaluating recall, precision, and F1-score


The accuracy scores and classification reports for the highest-performing models can be seen below. 


![Accuracy](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/Accuracy.png)



The evaluation metrics for several other models using Pycaret are provided in the below diagram. From this, we can observe that Extra Trees Classifier and Extreme Gradient Boosting have the best accuracy.




![allmodelaccuracy](https://github.com/tejasayya/Alzheimer-s-Disease-Analysis/blob/main/assets/allmodelaccuracy.png)




# Conclusion / Results


In this project, we explored various machine learning models to predict [target variable - Diagnosis] using the [Alzheimer's Disease] dataset. After preparing the data, we experimented with different models including [Random Forest], [Logistic Regression], and [K-Nearest Neighbors] using PyCaret.

The best performing model was [Extreme Gradient Boosting], which achieved an accuracy of [96.08%] and an AUC-ROC score of [98.96%]. This model was able to Identifiy top 10 features that causes Alzheimer's

  1. Functional Assessment
  2. ADL
  3. MMSE

Some of the key insights from our analysis include:

  - the ADL, MMSE, and FunctionalAssessment are the most important features
  - Features like CholesterolTotal, SleepQuality, and BMI have much lower importance scores

Overall, the project demonstrated that the best model for predicting Alzheimer's Desease based on the given dataset is Gradient Boosting Classifier, achieving up to 96.08% accuracy.




# Known Issues:

Despite the successes of the project, several issues were encountered:

  1. Data Quality:

   - The dataset contained outliers that could affect model performance.

  2. Model Limitations:

   - The models may not generalize well to unseen data due to overfitting on the training set.
   - The dataset was imbalanced, which might have affected the performance of certain models, especially in predicting minority classes.

  3. Bias and Variance:

   - The models might have inherent biases due to the nature of the dataset.
   - There was a trade-off between bias and variance in the models, affecting their predictive power.

  4. Feature Selection:

   - Some features may not have contributed significantly to the model, leading to potential overfitting.
   - There were challenges in selecting the most relevant features due to multicollinearity.

    
Future work should address these issues by collecting more data, using different imputation methods, pre-processing data rigorously applying more robust cross-validation techniques, and exploring advanced feature selection methods.

