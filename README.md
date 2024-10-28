# AI-IN-ONCOLOGY
## Overview
This project aims to explore machine learning applications in oncology, specifically for predicting outcomes in multiple myeloma patients. By utilizing various machine learning models, the project seeks to identify significant predictors of survival and improve decision-making processes in cancer treatment.

## Key Components
### 1. Data Import
The project begins by importing a dataset from an Excel file containing multiple myeloma patient data.

### 2. Data Preprocessing
- Missing Data Check: The dataset is analyzed for missing values across critical features such as ECOG index, regimen, age, and others.
- Data Type Verification: It ensures that variables are of the correct data type (numeric or factor).
- Type Conversion: Necessary conversions are made to prepare the data for analysis, including converting factors to numeric types and vice versa.
- Duplicate Check: The dataset is checked for duplicate entries.

### 3. Data Preparation
- The dataset is refined to include only relevant variables for the analysis, such as age, sex, ECOG index, and others.
Training and Test Set Creation
- The dataset is split into training and testing sets using a 75/25 ratio to ensure proper model validation.
- Feature Scaling
Numeric features are scaled to improve model performance.

### 4. Model Development
- Logistic Regression: A logistic regression model is fitted to the training data, and predictions are made on the test set. A confusion matrix is generated to evaluate prediction accuracy.
- Decision Tree Classifier: A decision tree model is created and evaluated similarly, providing another classification approach.
- Random Forest Classifier: A random forest model is trained and tested, offering robustness against overfitting.

### 5. Model Evaluation
The models' predictions are evaluated using confusion matrices, focusing on metrics like accuracy, false positives, and false negatives.
Insights and Recommendations
The project identifies challenges in prediction accuracy and suggests various techniques for improvement, such as checking for multicollinearity, applying regularization, and performing feature selection.
Cross-validation methods are proposed for better model performance assessment.
The need for alternative regression techniques is noted to handle issues like perfect separation.

### 6. Conclusion
This repository showcases a comprehensive approach to applying machine learning in oncology, specifically focusing on predicting outcomes in multiple myeloma patients. The project employs various models and evaluation techniques, highlighting the potential for AI to support clinical decisions in cancer treatment. The ongoing analysis and model refinement will contribute to the development of reliable predictive tools in healthcare.
