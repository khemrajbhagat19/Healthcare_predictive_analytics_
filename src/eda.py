import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load Dataset
df = pd.read_csv("data/diabetes.csv")

# First 5 Rows
print("First 5 Rows")
print(df.head())

# Shape
print("\nShape of Dataset")
print(df.shape)

# Columns
print("\nColumn Names")
print(df.columns)

# Information
print("\nDataset Information")
print(df.info())

# Statistics
print("\nStatistical Summary")
print(df.describe())

# Missing Values
print("\nMissing Values")
print(df.isnull().sum())
print("\nZero Values")

print("Glucose :", (df["Glucose"] == 0).sum())
print("BloodPressure :", (df["BloodPressure"] == 0).sum())
print("SkinThickness :", (df["SkinThickness"] == 0).sum())
print("Insulin :", (df["Insulin"] == 0).sum())
print("BMI :", (df["BMI"] == 0).sum())
import numpy as np

columns = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

df[columns] = df[columns].replace(0, np.nan)
print("\nMissing Values After Replacing Zero")

print(df.isnull().sum())
df.fillna(df.mean(), inplace=True)
print("\nMissing Values After Replacing Zero")

print(df.isnull().sum())
print("\nOutcome Count")
print(df["Outcome"].value_counts())
plt.figure(figsize=(6,4))

df["Outcome"].value_counts().plot(kind="bar")

plt.title("Diabetes Count")

plt.xlabel("Outcome")

plt.ylabel("Number of Patients")

plt.show()
plt.figure(figsize=(8,5))

plt.hist(df["Age"], bins=15)

plt.title("Age Distribution")

plt.xlabel("Age")

plt.ylabel("Number of Patients")

plt.show()
plt.figure(figsize=(8,5))

plt.hist(df["BMI"], bins=15)

plt.title("BMI Distribution")

plt.xlabel("BMI")

plt.ylabel("Frequency")

plt.show()
plt.figure(figsize=(8,5))

plt.hist(df["Glucose"], bins=15)

plt.title("Glucose Distribution")

plt.xlabel("Glucose")

plt.ylabel("Frequency")

plt.show()