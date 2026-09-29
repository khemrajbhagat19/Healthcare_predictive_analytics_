import pandas as pd
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# -----------------------------
# Load Dataset
# -----------------------------

df = pd.read_csv("data/diabetes.csv")


# -----------------------------
# Clean Invalid Values
# -----------------------------

columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_with_zero:
    df[column] = df[column].replace(0, df[column].median())


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
# Load Scaler
# -----------------------------

scaler = joblib.load("models/scaler.pkl")


X_train_scaled = scaler.transform(X_train)

X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Logistic Regression
# -----------------------------

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_scaled,
    y_train
)

logistic_prediction = logistic_model.predict(
    X_test_scaled
)

logistic_accuracy = accuracy_score(
    y_test,
    logistic_prediction
)


# -----------------------------
# Decision Tree
# -----------------------------

decision_tree = DecisionTreeClassifier(
    random_state=42
)

decision_tree.fit(
    X_train_scaled,
    y_train
)

decision_prediction = decision_tree.predict(
    X_test_scaled
)

decision_accuracy = accuracy_score(
    y_test,
    decision_prediction
)


# -----------------------------
# Random Forest
# -----------------------------

rf_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

rf_model.fit(
    X_train_scaled,
    y_train
)

rf_prediction = rf_model.predict(
    X_test_scaled
)

rf_accuracy = accuracy_score(
    y_test,
    rf_prediction
)


# -----------------------------
# Model Comparison
# -----------------------------

print("\n========== MODEL ACCURACY ==========")

print(
    "Logistic Regression:",
    round(logistic_accuracy * 100, 2),
    "%"
)

print(
    "Decision Tree:",
    round(decision_accuracy * 100, 2),
    "%"
)

print(
    "Random Forest:",
    round(rf_accuracy * 100, 2),
    "%"
)


# -----------------------------
# Save Random Forest Model
# -----------------------------

joblib.dump(
    rf_model,
    "models/diabetes_model.pkl"
)

print("\nRandom Forest model saved successfully!")


# -----------------------------
# Feature Importance
# -----------------------------

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf_model.feature_importances_
})

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print("\n========== FEATURE IMPORTANCE ==========")

print(importance.to_string(index=False))

importance.to_csv(
    "data/feature_importance.csv",
    index=False
)

print("\nFeature importance saved.")