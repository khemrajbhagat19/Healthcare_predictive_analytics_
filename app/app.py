import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from streamlit_option_menu import option_menu
from reportlab.pdfgen import canvas


# ============================================================
# LOAD MODEL
# ============================================================

model = joblib.load(
    "models/diabetes_model.pkl"
)

scaler = joblib.load(
    "models/scaler.pkl"
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)


# ============================================================
# CUSTOM CSS - MEDICAL THEME
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #f5f9fc;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .main-title {
        font-size: 42px;
        font-weight: 700;
        color: #0b5ed7;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        color: #5f6b7a;
        margin-bottom: 25px;
    }

    .info-card {
        padding: 20px;
        border-radius: 15px;
        background-color: white;
        border: 1px solid #e3eaf0;
        box-shadow: 0px 4px 15px rgba(0,0,0,0.05);
        margin-bottom: 20px;
    }

    .disclaimer {
        padding: 15px;
        border-radius: 10px;
        background-color: #fff8e1;
        border: 1px solid #ffe082;
        color: #6d4c00;
        margin-top: 25px;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        "## 🏥 Healthcare Analytics"
    )

    st.markdown("---")

    selected = option_menu(

        menu_title=None,

        options=[
            "Prediction",
            "Model Info",
            "Dataset",
            "About"
        ],

        icons=[
            "activity",
            "cpu",
            "database",
            "person"
        ],

        default_index=0
    )

    st.markdown("---")

    st.success(
        "Model: Random Forest"
    )

    st.info(
        "Dataset: PIMA Indians Diabetes"
    )

    st.markdown("---")

    st.markdown(
        """
        **Developer**

        Khemraj Bhagat

        B.Tech CSE  
        Data Science
        """
    )


# ============================================================
# PDF REPORT FUNCTION
# ============================================================

def create_pdf(
    result,
    confidence,
    patient_data
):

    pdf_file = "Diabetes_Report.pdf"

    c = canvas.Canvas(
        pdf_file
    )

    c.setTitle(
        "Diabetes Prediction Report"
    )

    c.setFont(
        "Helvetica-Bold",
        20
    )

    c.drawString(
        140,
        800,
        "Diabetes Prediction Report"
    )

    c.setFont(
        "Helvetica",
        12
    )

    y = 750

    c.drawString(
        50,
        y,
        f"Prediction: {result}"
    )

    y -= 25

    c.drawString(
        50,
        y,
        f"Confidence: {confidence:.2f}%"
    )

    y -= 40

    c.setFont(
        "Helvetica-Bold",
        14
    )

    c.drawString(
        50,
        y,
        "Patient Parameters"
    )

    y -= 25

    c.setFont(
        "Helvetica",
        11
    )

    for key, value in patient_data.items():

        c.drawString(
            50,
            y,
            f"{key}: {value}"
        )

        y -= 20

    y -= 20

    c.drawString(
        50,
        y,
        "Note: This prediction is for educational"
    )

    y -= 15

    c.drawString(
        50,
        y,
        "and analytical purposes only."
    )

    c.save()

    return pdf_file


# ============================================================
# PREDICTION PAGE
# ============================================================

if selected == "Prediction":

    st.markdown(
        '<div class="main-title">🩺 Diabetes Prediction System</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="subtitle">Machine Learning based healthcare risk analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        "### 👤 Enter Patient Details"
    )


    # --------------------------------------------------------
    # INPUTS
    # --------------------------------------------------------

    col1, col2 = st.columns(2)


    with col1:

        pregnancies = st.number_input(
            "Pregnancies",
            min_value=0,
            max_value=20,
            value=1
        )

        glucose = st.number_input(
            "Glucose Level",
            min_value=0.0,
            value=120.0
        )

        blood_pressure = st.number_input(
            "Blood Pressure",
            min_value=0.0,
            value=70.0
        )

        skin_thickness = st.number_input(
            "Skin Thickness",
            min_value=0.0,
            value=20.0
        )


    with col2:

        insulin = st.number_input(
            "Insulin",
            min_value=0.0,
            value=80.0
        )

        bmi = st.number_input(
            "BMI",
            min_value=0.0,
            value=25.0
        )

        dpf = st.number_input(
            "Diabetes Pedigree Function",
            min_value=0.0,
            value=0.5
        )

        age = st.number_input(
            "Age",
            min_value=1,
            max_value=120,
            value=30
        )


    st.markdown("")


    # --------------------------------------------------------
    # PREDICT BUTTON
    # --------------------------------------------------------

    predict_button = st.button(
        "🔍 Predict Diabetes",
        use_container_width=True
    )


    if predict_button:

        # ----------------------------------------------------
        # CREATE PATIENT DATA
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # SCALE DATA
        # ----------------------------------------------------

        patient_scaled = scaler.transform(
            patient
        )


        # ----------------------------------------------------
        # PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(
            patient_scaled
        )


        probability = model.predict_proba(
            patient_scaled
        )


        predicted_class = prediction[0]


        if predicted_class == 1:

            result = "Diabetic"

            confidence = (
                probability[0][1] * 100
            )

        else:

            result = "Not Diabetic"

            confidence = (
                probability[0][0] * 100
            )


        # ----------------------------------------------------
        # RESULT
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📊 Prediction Result"
        )


        if predicted_class == 1:

            st.error(
                "🔴 Patient is likely Diabetic"
            )

        else:

            st.success(
                "🟢 Patient is likely Not Diabetic"
            )


        # ----------------------------------------------------
        # METRICS
        # ----------------------------------------------------

        metric1, metric2 = st.columns(2)


        with metric1:

            st.metric(
                "Prediction",
                result
            )


        with metric2:

            st.metric(
                "Confidence",
                f"{confidence:.2f}%"
            )


        st.progress(
            min(confidence / 100, 1.0)
        )


        # ----------------------------------------------------
        # PIE CHART
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "📈 Prediction Probability"
        )


        fig, ax = plt.subplots(
            figsize=(6, 5)
        )


        labels = [
            "Not Diabetic",
            "Diabetic"
        ]


        sizes = [
            probability[0][0],
            probability[0][1]
        ]


        ax.pie(
            sizes,
            labels=labels,
            autopct="%1.1f%%",
            startangle=90
        )


        ax.axis("equal")


        st.pyplot(
            fig
        )


        # ----------------------------------------------------
        # FEATURE IMPORTANCE
        # ----------------------------------------------------

        st.markdown("---")

        st.subheader(
            "🔎 Feature Importance"
        )


        feature_file = (
            "data/feature_importance.csv"
        )


        try:

            importance_df = pd.read_csv(
                feature_file
            )


            fig, ax = plt.subplots(
                figsize=(8, 5)
            )


            ax.barh(
                importance_df["Feature"],
                importance_df["Importance"]
            )


            ax.set_xlabel(
                "Importance"
            )


            ax.set_title(
                "Random Forest Feature Importance"
            )


            ax.invert_yaxis()


            st.pyplot(
                fig
            )


        except FileNotFoundError:

            st.info(
                "Feature importance file not found. "
                "Run train_model.py first."
            )


        # ----------------------------------------------------
        # PDF REPORT
        # ----------------------------------------------------

        patient_data = {

            "Pregnancies": pregnancies,

            "Glucose": glucose,

            "Blood Pressure": blood_pressure,

            "Skin Thickness": skin_thickness,

            "Insulin": insulin,

            "BMI": bmi,

            "Diabetes Pedigree Function": dpf,

            "Age": age

        }


        pdf_file = create_pdf(

            result,

            confidence,

            patient_data

        )


        with open(
            pdf_file,
            "rb"
        ) as file:

            st.download_button(

                label="📄 Download PDF Report",

                data=file,

                file_name="Diabetes_Report.pdf",

                mime="application/pdf",

                use_container_width=True

            )


    # --------------------------------------------------------
    # DISCLAIMER
    # --------------------------------------------------------

    st.markdown(
        """
        <div class="disclaimer">
        ⚠️ <b>Important:</b> This application is an educational
        machine-learning project and should not be used as a
        substitute for professional medical diagnosis or treatment.
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# MODEL INFO PAGE
# ============================================================

elif selected == "Model Info":

    st.title(
        "🤖 Model Information"
    )

    st.write(
        "This project uses multiple classification algorithms."
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.info(
            """
            ### Logistic Regression

            Used as a baseline
            classification model.
            """
        )


    with col2:

        st.info(
            """
            ### Decision Tree

            Tree-based classification
            model.
            """
        )


    with col3:

        st.info(
            """
            ### Random Forest

            Ensemble model used for
            final prediction.
            """
        )


# ============================================================
# DATASET PAGE
# ============================================================

elif selected == "Dataset":

    st.title(
        "📊 Dataset Information"
    )

    st.write(
        """
        The project uses the PIMA Indians Diabetes
        dataset for educational machine learning analysis.
        """
    )


    try:

        df = pd.read_csv(
            "data/diabetes.csv"
        )

        st.write(
            "Dataset Shape:",
            df.shape
        )

        st.dataframe(
            df.head(20),
            use_container_width=True
        )

    except FileNotFoundError:

        st.error(
            "diabetes.csv not found in data folder."
        )


# ============================================================
# ABOUT PAGE
# ============================================================

elif selected == "About":

    st.title(
        "ℹ️ About Project"
    )

    st.write(
        """
        ### Healthcare Predictive Analytics

        This project demonstrates an end-to-end
        machine learning workflow for diabetes
        prediction.

        The workflow includes:

        • Data preprocessing

        • Exploratory data analysis

        • Feature scaling

        • Machine learning classification

        • Model evaluation

        • Feature importance

        • Streamlit deployment interface

        • PDF report generation
        """
    )

    st.markdown("---")

    st.write(
        "**Developer:** Khemraj Bhagat"
    )

    st.write(
        "**Program:** B.Tech CSE (Data Science)"
    )