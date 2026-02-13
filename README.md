Heart Disease Prediction Project
1. 🔍 Project Overview
This project aims to analyze clinical data and develop a machine learning model to predict the presence of heart disease in patients. Using the Kaggle Heart Disease dataset, I implemented a classification pipeline to classify patients as either having or not having heart disease based on 13 medical attributes.
Goal: To assist healthcare professionals in early detection by achieving a high roc-aoc rate which shows a strong ability to distinguish between classes (positive and negative cases).

2. 📊 Dataset Information
Source: Kaggle (originally UCI Cleveland dataset).
Observations: 629,999 entries.
Key Features:
Age: Age of the patient.
Sex: 1 = Male; 0 = Female.
Chest pain type: Chest pain type (4 types).
BP: Blood Pressure in mm Hg
Cholesterol: Serum cholesterol in mg/dl.
FBS over 120: Fasting blood sugar > 120 mg/dl.
EKG results: Electrocardiogram
Max HR: Maximum heart rate in bpm
ST depression
Slope of ST
Number of vessels fluro
Thallium
Exercise angina: Exercise-induced angina.
Heart Disease: 0 = Absence, 1 = Presence.

3. 🛠️ Methodology & Techniques
Exploratory Data Analysis (EDA): Visualized feature distributions and correlations to identify key drivers (e.g., cp, max heart rate, chest pain).
Data Preprocessing: Scaled numerical data (StandardScaler), and encoded categorical variables, select features, group features.
Model Building: Evaluated multiple classification algorithms:
FastAI 
eXtreme Gradient Boosting Classifier(XGB Classifier)

Evaluation Metrics: Primarily focused on ROC-AOC.

4. 📈 Results & Key Findings
The XGBClassifier achieved an accuracy of 88.7% and a ROC-AOC of 95.51% on the test set.
Key Predictors: Thallium, Max HR, High cholesterol, low maximum heart rate, and asymptomatic chest pain are strong indicators of heart disease.
See EDA plots in the notebook for visual insights.

5. ✉️ Contact
Author: Engr. Obinna Dijeh
LinkedIn: https://www.linkedin.com/in/dijeh-obinna/
Kaggle: https://www.kaggle.com/dijeh08