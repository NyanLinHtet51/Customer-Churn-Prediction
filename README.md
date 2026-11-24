
# Customer Churn Prediction

## 1. Project Overview

Customer Churn Prediction is a Machine Learning project that predicts whether a telecommunications customer is likely to leave the company.

The project uses customer information and service-related features to train and evaluate three Machine Learning models. A Streamlit web application allows users to enter customer details, select a model, and view churn predictions and risk levels.

## 2. Project Objectives

- Analyze customer churn patterns.
- Identify factors associated with customer churn.
- Build and compare different Machine Learning models.
- Predict customer churn and estimate churn probability.
- Provide business recommendations to help reduce customer churn.

## 3. Dataset

**Dataset:** IBM Telco Customer Churn

**Source:** https://github.com/IBM/telco-customer-churn-on-icp4d

- Total records: 7,043 customers
- Original columns: 21
- Target variable: `Churn`
- Target classes: `Yes` and `No`

The dataset contains customer demographics, subscribed services, contract information, tenure, and billing details.

To run this project, download the CSV dataset and place it here:

`data/Telco-Customer-Churn.csv`

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Joblib
- Streamlit

## 5. Data Preprocessing

The following preprocessing techniques were applied:

- Removed `customerID` from model features.
- Converted `TotalCharges` into numeric format.
- Handled missing values using SimpleImputer.
- Applied StandardScaler to numerical features.
- Applied OneHotEncoder to categorical features.
- Split the dataset into 80% training data and 20% testing data.

## 6. Machine Learning Models

Three classification models were trained and evaluated:

1. Logistic Regression
2. Decision Tree
3. Random Forest

## 7. Model Evaluation Results

The models were evaluated using Accuracy, Precision, Recall, and F1-score.

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---:|---:|---:|---:|
| Logistic Regression | 80.55% | 65.72% | 55.88% | 60.40% |
| Decision Tree | 75.59% | 52.79% | 75.94% | 62.28% |
| Random Forest | 78.14% | 61.70% | 46.52% | 53.05% |

**Best Model by F1-Score: Decision Tree (62.28%)**

The Decision Tree achieved the highest Recall and F1-score, making it useful when identifying more customers who may churn is a priority.

## 8. Streamlit Web Application

The application provides:

- Customer information input form.
- Machine Learning model selection.
- Customer churn prediction.
- Churn probability percentage.
- Low, Medium, and High risk classification.
- Business recommendations based on risk level.
- Prediction history.
- Model performance metrics.
- Confusion matrix visualization.
- Comparison of all three models' predictions.

## 9. Project Structure

```text
Customer-Churn-Prediction/
│
├── main.py
├── app.py
├── README.md
├── requirements.txt
├── .gitignore
│
└── data/
    └── Telco-Customer-Churn.csv
```

The generated `model_comparison.pkl` file is created when the training script runs. It is not included in this repository.

## 10. How to Run the Project

### Step 1: Clone the repository

```bash
git clone https://github.com/NyanLinHtet51/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction
```

### Step 2: Create a virtual environment

```bash
python -m venv .venv
```

### Step 3: Activate the virtual environment

Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### Step 4: Install dependencies

```bash
python -m pip install -r requirements.txt
```

### Step 5: Download the dataset

Download the IBM Telco Customer Churn CSV and place it inside the `data` folder using this filename:

`Telco-Customer-Churn.csv`

### Step 6: Train the models

```bash
python main.py
```

This will train the models, evaluate their performance, and generate `model_comparison.pkl`.

### Step 7: Run the Streamlit application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal to use the application.

## 11. Conclusion

This project demonstrates how Machine Learning can help telecommunications companies identify customers who are likely to leave their services.

By comparing three classification models and developing an interactive Streamlit application, the project provides a practical approach to customer churn prediction and supports data-driven customer retention strategies.

---

**Project Type:** Machine Learning / Classification

**Application:** Streamlit Web Application
