# ==========================================
# CUSTOMER CHURN PREDICTION PROJECT
# ==========================================

# 1. IMPORT LIBRARIES

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler


from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

# ==========================================
# 2. LOAD DATASET
# ==========================================

df = pd.read_csv("data/Telco-Customer-Churn.csv")

print("Dataset Shape:", df.shape)

# ==========================================
# 3. DATA CLEANING
# ==========================================

# Convert TotalCharges into numeric values
# Invalid or blank values become NaN

df["TotalCharges"] = pd.to_numeric(
    df["TotalCharges"],
    errors="coerce"
)

print("\nMissing Values:")
print(df.isnull().sum())

# ==========================================
# 4. SEPARATE FEATURES AND TARGET
# ==========================================

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

numerical_features = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str"]
).columns.tolist()

print("\nNumerical Features:", numerical_features)
print("Categorical Features:", categorical_features)

# ==========================================
# 5. TRAIN-TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Shape:", X_train.shape)
print("Testing Shape:", X_test.shape)

# ==========================================
# 6. DATA PREPROCESSING
# ==========================================

# Numerical data preprocessing

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical data preprocessing

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore"))
])

# Combine both preprocessing pipelines

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numerical_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)

# ==========================================
# 7. CREATE MACHINE LEARNING MODELS
# ==========================================


models = {
    "Logistic Regression": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", LogisticRegression(max_iter=1000))
    ]),

    "Decision Tree": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(
            max_depth=5,
            min_samples_split=20,
            min_samples_leaf=10,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Random Forest": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "SVM": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", SVC(
            kernel="rbf",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42
        ))
    ]),

    "Gradient Boosting": Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=42
        ))
    ])
}

# ==========================================
# 8. TRAIN AND EVALUATE MODELS
# ==========================================

model_results = {}
confusion_matrices = {}

for name, model in models.items():

    print(f"\n========== {name} ==========")

    # Train model
    model.fit(X_train, y_train)

    # Make predictions
    predictions = model.predict(X_test)

    # Calculate evaluation metrics
    accuracy = accuracy_score(y_test, predictions) * 100

    precision = precision_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    ) * 100

    recall = recall_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    ) * 100

    f1 = f1_score(
        y_test, predictions, pos_label="Yes", zero_division=0
    ) * 100

    # Store results
    model_results[name] = {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1
    }

    # Store confusion matrix
    confusion_matrices[name] = confusion_matrix(
        y_test,
        predictions,
        labels=["No", "Yes"]
    )

    # Display results
    print(f"Accuracy: {accuracy:.2f}%")
    print(f"Precision: {precision:.2f}%")
    print(f"Recall: {recall:.2f}%")
    print(f"F1-Score: {f1:.2f}%")

    print("\nClassification Report:")
    print(classification_report(y_test, predictions))

# ==========================================
# 9. MODEL COMPARISON
# ==========================================

comparison_df = pd.DataFrame(model_results).T

best_model = comparison_df["F1-Score"].idxmax()

print("\n========== BEST MODEL ==========")
print("Best Model based on F1-Score:", best_model)

print("\n========== MODEL COMPARISON ==========")
print(comparison_df.round(2))

# ==========================================
# 10. CONFUSION MATRICES
# ==========================================

for name, cm in confusion_matrices.items():

    plt.figure(figsize=(6, 5))

    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["No Churn", "Churn"],
        yticklabels=["No Churn", "Churn"]
    )

    plt.title(f"{name} - Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.tight_layout()
    plt.show()

# ==========================================
# 11. MODEL PERFORMANCE COMPARISON CHART
# ==========================================

comparison_df.plot(
    kind="bar",
    figsize=(10, 6)
)

plt.title("Machine Learning Model Comparison")
plt.xlabel("Machine Learning Models")
plt.ylabel("Performance (%)")
plt.xticks(rotation=15)
plt.ylim(0, 100)
plt.legend(loc="best")
plt.tight_layout()
plt.show()
# ==========================================
# 13. SAVE MODELS AND RESULTS
# ==========================================

# Save all five models and their evaluation results

model_comparison = {
    "models": models,
    "results": model_results,
    "confusion_matrices": confusion_matrices,
    "y_test": y_test.tolist()
}

joblib.dump(
    model_comparison,
    "model_comparison.pkl"
)

print("\n==========================================")
print("ALL MODELS TRAINED SUCCESSFULLY!")
print("ALL MODELS AND RESULTS SAVED!")
print("==========================================")
