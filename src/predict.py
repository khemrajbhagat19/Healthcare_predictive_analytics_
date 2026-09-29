import joblib
import pandas as pd


# -----------------------------
# Load Model and Scaler
# -----------------------------

model = joblib.load(
    "models/diabetes_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


print("\n====================================")
print("       DIABETES PREDICTION")
print("====================================")


# -----------------------------
# User Input
# -----------------------------

pregnancies = float(
    input("Pregnancies: ")
)

glucose = float(
    input("Glucose: ")
)

blood_pressure = float(
    input("Blood Pressure: ")
)

skin_thickness = float(
    input("Skin Thickness: ")
)

insulin = float(
    input("Insulin: ")
)

bmi = float(
    input("BMI: ")
)

dpf = float(
    input("Diabetes Pedigree Function: ")
)

age = float(
    input("Age: ")
)


# -----------------------------
# Create DataFrame
# -----------------------------

patient = pd.DataFrame([{

    "Pregnancies": pregnancies,

    "Glucose": glucose,

    "BloodPressure": blood_pressure,

    "SkinThickness": skin_thickness,

    "Insulin": insulin,

    "BMI": bmi,

    "DiabetesPedigreeFunction": dpf,

    "Age": age

}])


# -----------------------------
# Scale Input
# -----------------------------

patient_scaled = scaler.transform(
    patient
)


# -----------------------------
# Prediction
# -----------------------------

prediction = model.predict(
    patient_scaled
)


probability = model.predict_proba(
    patient_scaled
)


confidence = probability.max() * 100


# -----------------------------
# Result
# -----------------------------

print("\n====================================")

if prediction[0] == 1:

    print("Prediction : Diabetic")

else:

    print("Prediction : Not Diabetic")


print(
    f"Confidence : {confidence:.2f}%"
)

print("====================================")