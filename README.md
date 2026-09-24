# ML-Based Student Academic Performance Prediction System

## Machine Learning-Based Early Warning and Academic Risk Prediction

This project presents a machine learning-based system for predicting
student academic performance and identifying students who may be at risk
of poor academic performance.

The system uses the TabPFN model for tabular-data prediction and provides
risk scoring, prediction explanations, academic trend analysis,
intervention recommendations, and what-if analysis.

---

## Key Features

- Student academic performance prediction
- At-Risk student identification
- TabPFN-based classification
- Risk score visualization
- Explainable predictions
- Academic trend analysis
- Intervention recommendations
- What-if analysis
- Teacher dashboard
- Web-based interface
- Docker support

---

## Technology Stack

### Machine Learning

- Python
- Pandas
- NumPy
- Scikit-learn
- TabPFN
- Joblib

### Web Development

- Flask
- HTML
- CSS
- JavaScript

### Deployment / Development

- Docker
- Git
- GitHub

---

## System Workflow

```text

Student Data
      ↓
Data Preprocessing
      ↓
Feature Engineering
      ↓
19 Engineered Features
      ↓
TabPFN Model
      ↓
Performance Prediction
      ↓
Risk Score
      ↓
Risk Level
      ↓
Explanation
      ↓
Recommendations
      ↓
What-If Analysis / Dashboard

```
---

## Dataset

The project uses the UCI Student Performance dataset.

Dataset size:

- 395 student records
- Original dataset contains 33 attributes
- 19 engineered features are used by the final model

The target is defined as:

- Good Performance: G3 >= 10
- At Risk: G3 < 10

The final G3 value is not used as an input feature to avoid target
leakage.

---

## Machine Learning Model

The project uses:

**TabPFN — Tabular Prior-Data Fitted Network**

The model configuration was evaluated using 5-fold stratified
cross-validation.

Different TabPFN ensemble configurations were tested:

- n_estimators = 4
- n_estimators = 8
- n_estimators = 16

The selected configuration was:

**n_estimators = 16**

---

## Model Performance

### Holdout Test Results

| Metric | Result |
|---|---:|
| Accuracy | 89.87% |
| Precision | 82.14% |
| Recall | 88.46% |
| F1 Score | 85.19% |

Cross-validation results are also stored in:

`artifacts/model_results.csv`

---

## Application Features

### 1. Student Prediction

The system accepts student academic and related information and
predicts whether the student is at risk.

### 2. Risk Score

The system displays an individual student's predicted risk probability.

### 3. Explainable Prediction

The application identifies important academic and behavioral factors
associated with the prediction.

### 4. Recommendations

The system provides practical intervention suggestions based on
identified risk factors.

### 5. What-If Analysis

Users can modify selected student attributes and observe how the
prediction changes.

### 6. Teacher Dashboard

The dashboard provides an overview of student predictions and risk
levels.

---

## Project Structure

```text
ML-Student-Academic-Performance-Prediction/
├── data/
├── src/
├── artifacts/
├── templates/
├── static/
├── reports/
├── screenshots/
├── application.py
├── train.py
├── requirements.txt
├── Dockerfile
└── README.md
```

## Team Members
| Id | Name |
|---|---:|
| 2420030285 | Vignesh Reddy |
| 2420030404 | Karthik |
| 2420080062 | Nishanth |
| 2420090088 | Niranjan Reddy |

## Supervisor
Dr. K. Swanthana
