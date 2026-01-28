# Heart Disease Risk Classification with Logistic Regression

## Project Overview

This repository presents an end-to-end implementation of **logistic regression built from scratch** to classify the presence of heart disease. The main goal is to understand the mathematical foundations of the algorithm rather than relying on high-level ML libraries.

The project covers the full workflow, including:

- Exploratory analysis of medical data
- Manual implementation of logistic regression
- Model evaluation and comparison
- Regularization techniques to improve generalization
- Initial exploration of cloud deployment options

## Dataset Information

### Origin

**UCI Heart Disease Dataset (via Kaggle)**  
https://www.kaggle.com/datasets/neurocipher/heartdisease

### General Characteristics

- **Number of records**: 303 patients
- **Total features**: 14 medical attributes
- **Target variable**: Binary (heart disease: yes / no)
- **Class distribution**: ~55% positive cases
- **Missing data**: None detected
- **Patient age**: 29 to 77 years
- **Cholesterol values**: 112–564 mg/dL

### Features Used for Training (8)

The following features were selected after exploratory analysis:

1. Age
2. Serum Cholesterol
3. Resting Blood Pressure
4. Maximum Heart Rate Achieved
5. ST Depression (exercise-induced)
6. Number of Major Vessels (fluoroscopy)
7. Chest Pain Category
8. Exercise-Induced Angina

## Model Evaluation

### Logistic Regression (No Regularization)

- **Training Accuracy**: 86.32%
- **Test Accuracy**: 85.71%
- **F1 Score (Test)**: 0.8571
- **Optimization**: Gradient descent
- **Hyperparameters**:
  - Learning rate (α): 0.01
  - Iterations: 1000

### Logistic Regression with L2 Regularization

- **Regularization parameter (λ)**: 0.01
- **Training Accuracy**: 86.32%
- **Test Accuracy**: 86.81%
- **F1 Score (Test)**: 0.8696
- **Weight norm reduction**: ~15% compared to the unregularized model

### Feature Impact

Based on the magnitude of learned coefficients, the most relevant predictors were:

1. ST Depression
2. Number of Vessels (Fluoroscopy)
3. Exercise-Induce
