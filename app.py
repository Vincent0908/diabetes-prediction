import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import streamlit as st
from tensorflow import keras

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)


st.set_page_config(
    page_title="Diabetes Prediction",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1e3a5f;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .result-positive {
        background-color: #fef2f2;
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .result-negative {
        background-color: #f0fdf4;
        border: 2px solid #22c55e;
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
    }
    .stButton > button {
        background-color: #1e3a5f;
        color: white;
        border-radius: 8px;
        padding: 0.6rem 2rem;
        font-size: 1rem;
        font-weight: 600;
        border: none;
        width: 100%;
        margin-top: 1rem;
    }
    .stButton > button:hover {
        background-color: #2563eb;
    }
    .section-title {
        font-size: 1.15rem;
        font-weight: 600;
        color: #1e3a5f;
        border-bottom: 2px solid #e2e8f0;
        padding-bottom: 0.4rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_models():
    models_dir = os.path.join(BASE_DIR, 'models')
    try:
        lr_model = joblib.load(os.path.join(models_dir, 'logistic_model.pkl'))
        nn_model = keras.models.load_model(os.path.join(models_dir, 'nn_model.keras'))
        scaler   = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        return lr_model, nn_model, scaler
    except FileNotFoundError as e:
        st.error(f"Model file not found: {e}. Run `python main.py` first.")
        st.stop()


@st.cache_data
def load_metrics():
    path = os.path.join(BASE_DIR, 'outputs', 'metrics_summary.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def preprocess_input(inputs: dict, scaler) -> np.ndarray:
    """Apply same feature engineering as training pipeline."""
    bmi        = inputs['BMI']
    insulin    = inputs['Insulin']
    glucose    = inputs['Glucose']
    age        = inputs['Age']
    pregnancies = inputs['Pregnancies']

    if bmi < 18.5:
        bmi_cat = 0
    elif bmi < 25:
        bmi_cat = 1
    elif bmi < 30:
        bmi_cat = 2
    else:
        bmi_cat = 3

    glucose_insulin_ratio = glucose / (insulin + 1)
    age_pregnancies = age * pregnancies

    feature_vector = np.array([[
        pregnancies, glucose, inputs['BloodPressure'],
        inputs['SkinThickness'], insulin, bmi,
        inputs['DiabetesPedigreeFunction'], age,
        bmi_cat, glucose_insulin_ratio, age_pregnancies
    ]])

    return scaler.transform(feature_vector)


def predict(model, X_scaled, is_keras: bool = False):
    if is_keras:
        prob = float(model.predict(X_scaled, verbose=0).flatten()[0])
    else:
        prob = float(model.predict_proba(X_scaled)[0][1])
    return int(prob >= 0.5), prob


def plot_feature_importance(lr_model, feature_names: list):
    coefficients = lr_model.coef_[0]
    sorted_idx = np.argsort(np.abs(coefficients))

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#ef4444' if c > 0 else '#3b82f6' for c in coefficients[sorted_idx]]
    ax.barh([feature_names[i] for i in sorted_idx], coefficients[sorted_idx],
            color=colors, edgecolor='white')
    ax.axvline(0, color='black', linewidth=0.8)
    ax.set_title('Feature Importance (LR Coefficients)', fontsize=12, fontweight='bold')
    ax.set_xlabel('Coefficient Value')
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    return fig


def main():
    lr_model, nn_model, scaler = load_models()
    metrics = load_metrics()

    feature_names = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age',
        'BMI_Category', 'Glucose_Insulin_Ratio', 'Age_Pregnancies'
    ]

    st.markdown('<p class="main-header">Diabetes Risk Prediction</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Pima Indians Diabetes Dataset — '
        'Logistic Regression & Neural Network</p>',
        unsafe_allow_html=True
    )

    tab1, tab2, tab3 = st.tabs(["Prediction", "Model Performance", "EDA Visualizations"])

    # Tab 1: Prediction
    with tab1:
        col_form, col_result = st.columns([1.1, 1])

        with col_form:
            st.markdown('<p class="section-title">Patient Clinical Data</p>', unsafe_allow_html=True)

            model_choice = st.radio("Select Model",
                                    ["Logistic Regression", "Neural Network"],
                                    horizontal=True)
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=3)
                glucose = st.number_input("Glucose (mg/dL)", min_value=0, max_value=250, value=117)
                blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=150, value=72)
                skin_thickness = st.number_input("Skin Thickness (mm)", min_value=0, max_value=110, value=29)

            with c2:
                insulin = st.number_input("Insulin (uU/mL)", min_value=0, max_value=900, value=125)
                bmi = st.number_input("BMI (kg/m2)", min_value=0.0, max_value=80.0, value=32.0, step=0.1)
                dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0,
                                      value=0.37, step=0.01)
                age = st.number_input("Age (years)", min_value=18, max_value=100, value=29)

            predict_btn = st.button("Run Prediction")

        with col_result:
            st.markdown('<p class="section-title">Prediction Result</p>', unsafe_allow_html=True)

            if predict_btn:
                inputs = {
                    'Pregnancies': pregnancies, 'Glucose': glucose,
                    'BloodPressure': blood_pressure, 'SkinThickness': skin_thickness,
                    'Insulin': insulin, 'BMI': bmi,
                    'DiabetesPedigreeFunction': dpf, 'Age': age
                }

                X_scaled = preprocess_input(inputs, scaler)
                is_keras = (model_choice == "Neural Network")
                model = nn_model if is_keras else lr_model
                pred, prob = predict(model, X_scaled, is_keras)

                if prob < 0.3:
                    risk_label, risk_color = "Low Risk", "#22c55e"
                elif prob < 0.6:
                    risk_label, risk_color = "Moderate Risk", "#f59e0b"
                else:
                    risk_label, risk_color = "High Risk", "#ef4444"

                result_class = "result-positive" if pred == 1 else "result-negative"
                result_text  = "Diabetes Detected" if pred == 1 else "No Diabetes Detected"

                st.markdown(f"""
                <div class="{result_class}">
                    <h3 style="margin:0.5rem 0; color:{'#ef4444' if pred==1 else '#16a34a'}">{result_text}</h3>
                    <p style="font-size:1.1rem; margin:0.3rem 0">
                        Probability: <strong>{prob:.1%}</strong>
                    </p>
                    <p style="font-size:0.95rem; color:{risk_color}; font-weight:600">{risk_label}</p>
                    <p style="font-size:0.8rem; color:#94a3b8; margin-top:0.5rem">Model: {model_choice}</p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Diabetes Probability**")
                st.progress(prob)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("**Input Summary**")
                summary_df = pd.DataFrame({'Feature': list(inputs.keys()), 'Value': list(inputs.values())})
                st.dataframe(summary_df, hide_index=True, use_container_width=True)

            else:
                st.info("Fill in patient data and click Run Prediction.")
                st.markdown("""
                #### About
                Predicts diabetes risk using clinical measurements from the
                Pima Indians Diabetes Dataset (UCI).

                **Input features:** Glucose, Insulin, BMI, Blood Pressure,
                Skin Thickness, Age, Pregnancies, Diabetes Pedigree Function.

                > This tool is for educational purposes only and is not a
                > substitute for professional medical advice.
                """)

    # Tab 2: Model Performance
    with tab2:
        st.markdown('<p class="section-title">Evaluation Metrics</p>', unsafe_allow_html=True)

        if metrics:
            lr_m = metrics['logistic_regression']
            nn_m = metrics['neural_network']

            metrics_df = pd.DataFrame({
                'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC'],
                'Logistic Regression': [lr_m['Accuracy'], lr_m['Precision'],
                                        lr_m['Recall'], lr_m['F1-Score'], lr_m['ROC-AUC']],
                'Neural Network': [nn_m['Accuracy'], nn_m['Precision'],
                                   nn_m['Recall'], nn_m['F1-Score'], nn_m['ROC-AUC']]
            })
            st.dataframe(
                metrics_df.style
                    .highlight_max(subset=['Logistic Regression', 'Neural Network'],
                                   color='#bbf7d0', axis=1)
                    .format({'Logistic Regression': '{:.4f}', 'Neural Network': '{:.4f}'}),
                use_container_width=True, hide_index=True
            )

        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Confusion Matrices**")
            cm_lr = os.path.join(BASE_DIR, 'outputs', 'cm_logistic.png')
            cm_nn = os.path.join(BASE_DIR, 'outputs', 'cm_nn.png')
            if os.path.exists(cm_lr):
                st.image(cm_lr, caption='Logistic Regression', use_column_width=True)
            if os.path.exists(cm_nn):
                st.image(cm_nn, caption='Neural Network', use_column_width=True)

        with col_b:
            roc_path = os.path.join(BASE_DIR, 'outputs', 'roc_comparison.png')
            if os.path.exists(roc_path):
                st.markdown("**ROC Curve Comparison**")
                st.image(roc_path, use_column_width=True)

            history_path = os.path.join(BASE_DIR, 'outputs', 'nn_training_history.png')
            if os.path.exists(history_path):
                st.markdown("**Neural Network Training History**")
                st.image(history_path, use_column_width=True)

        st.markdown("---")
        st.markdown('<p class="section-title">Feature Importance (Logistic Regression)</p>',
                    unsafe_allow_html=True)
        fig = plot_feature_importance(lr_model, feature_names)
        st.pyplot(fig, use_container_width=True)

    # Tab 3: EDA
    with tab3:
        st.markdown('<p class="section-title">Exploratory Data Analysis</p>', unsafe_allow_html=True)

        eda_plots = [
            ('eda_01_class_distribution.png',   'Class Distribution'),
            ('eda_02_feature_distributions.png', 'Feature Distributions by Class'),
            ('eda_03_correlation_heatmap.png',   'Correlation Heatmap'),
            ('eda_04_boxplots.png',              'Feature Boxplots'),
            ('eda_05_target_correlation.png',    'Feature Correlation with Target'),
        ]

        for filename, title in eda_plots:
            path = os.path.join(BASE_DIR, 'outputs', filename)
            if os.path.exists(path):
                st.markdown(f"**{title}**")
                st.image(path, use_column_width=True)
                st.markdown("---")

        data_path = os.path.join(BASE_DIR, 'data', 'diabetes.csv')
        if os.path.exists(data_path):
            st.markdown("**Raw Dataset Preview**")
            df = pd.read_csv(data_path)
            st.dataframe(df.head(10), use_container_width=True, hide_index=True)
            st.caption(f"{df.shape[0]} rows x {df.shape[1]} columns")


if __name__ == "__main__":
    main()
