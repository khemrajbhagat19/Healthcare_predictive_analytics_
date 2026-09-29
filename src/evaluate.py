import pandas as pd
import joblib

from sklearn.model_selection import train_test_split

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/diabetes.csv")


# -----------------------------
# Clean Data
# -----------------------------

columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_with_zero:
    df[column] = df[column].replace(
        0,
        df[column].median()
    )


# -----------------------------
# Features and Target
# -----------------------------

X = df.drop("Outcome", axis=1)

y = df["Outcome"]


# -----------------------------
# Train Test Split
# -----------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)


# -----------------------------
# Load Scaler and Model
# -----------------------------

scaler = joblib.load(
    "models/scaler.pkl"
)

model = joblib.load(
    "models/diabetes_model.pkl"
)


# -----------------------------
# Scale Data
# -----------------------------

X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(
    X_test_scaled
)


# -----------------------------
# Evaluation Metrics
# -----------------------------

accuracy = accuracy_score(
    y_test,
    prediction
)

precision = precision_score(
    y_test,
    prediction
)

recall = recall_score(
    y_test,
    prediction
)

f1 = f1_score(
    y_test,
    prediction
)


print("\n========== MODEL EVALUATION ==========")

print(
    "Accuracy:",
    round(accuracy * 100, 2),
    "%"
)

print(
    "Precision:",
    round(precision * 100, 2),
    "%"
)

print(
    "Recall:",
    round(recall * 100, 2),
    "%"
)

print(
    "F1 Score:",
    round(f1 * 100, 2),
    "%"
)


# -----------------------------
# Classification Report
# -----------------------------

print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        prediction
    )
)


# -----------------------------
# Confusion Matrix
# -----------------------------

print("\n========== CONFUSION MATRIX ==========")

print(
    confusion_matrix(
        y_test,
        prediction
    )
)