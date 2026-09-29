import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os


# -----------------------------
# Load Dataset
# -----------------------------

data_path = "data/diabetes.csv"

df = pd.read_csv(data_path)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)

print("\nFirst 5 Rows:")
print(df.head())


# -----------------------------
# Basic Information
# -----------------------------

print("\nDataset Information:")
print(df.info())

print("\nMissing Values:")
print(df.isnull().sum())


# -----------------------------
# Replace Invalid Zero Values
# -----------------------------

columns_with_zero = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for column in columns_with_zero:
    df[column] = df[column].replace(0, np.nan)


# -----------------------------
# Fill Missing Values
# -----------------------------

for column in columns_with_zero:
    df[column] = df[column].fillna(df[column].median())


print("\nMissing Values After Cleaning:")
print(df.isnull().sum())


# -----------------------------
# Separate Features and Target
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
# Feature Scaling
# -----------------------------

scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# -----------------------------
# Create Models Directory
# -----------------------------

os.makedirs("models", exist_ok=True)


# -----------------------------
# Save Scaler
# -----------------------------

joblib.dump(scaler, "models/scaler.pkl")


# -----------------------------
# Save Processed Data
# -----------------------------

train_data = pd.DataFrame(
    X_train_scaled,
    columns=X.columns
)

train_data["Outcome"] = y_train.values

test_data = pd.DataFrame(
    X_test_scaled,
    columns=X.columns
)

test_data["Outcome"] = y_test.values

train_data.to_csv(
    "data/train_processed.csv",
    index=False
)

test_data.to_csv(
    "data/test_processed.csv",
    index=False
)


print("\nPreprocessing Completed Successfully!")
print("Scaler saved to models/scaler.pkl")
print("Processed datasets saved.")