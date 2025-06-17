import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Formulir ini digunakan untuk memprediksi status mahasiswa berdasarkan data akademik dan demografis.")

model = joblib.load("model/gboost_model.joblib")
target_encoder = joblib.load("model/encoder_target.joblib")
pca_1 = joblib.load("model/pca_1.joblib")
pca_2 = joblib.load("model/pca_2.joblib")

numerical_pca_1 = [
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade'
]

numerical_pca_2 = [
    'Previous_qualification_grade', 'Admission_grade', 'Age_at_enrollment',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

categorical_columns = [
    "Marital_status", "Application_mode", "Course", "Previous_qualification",
    "Mothers_qualification", "Fathers_qualification", "Mothers_occupation",
    "Fathers_occupation", "Displaced", "Debtor",
    "Tuition_fees_up_to_date", "Gender", "Scholarship_holder",
]

scalers = {col: joblib.load(f"model/scaler_{col}.joblib") for col in numerical_pca_1 + numerical_pca_2}
encoders = {col: joblib.load(f"model/encoder_{col}.joblib") for col in categorical_columns}

def extract_int(value):
    try:
        return int(str(value).split(" - ")[0])
    except:
        return value

with st.form("form_prediksi"):
    st.subheader("📝 Data Mahasiswa")

    col_kiri, col_kanan = st.columns(2)
    inputs = {}

    with col_kiri:
        for col in categorical_columns:
            options = joblib.load(f"model/options_{col}.joblib")
            inputs[col] = st.selectbox(col.replace("_", " ").capitalize(), options)

    with col_kanan:
        for col in numerical_pca_1 + numerical_pca_2:
            label = col.replace("_", " ").capitalize()
            inputs[col] = st.number_input(label, step=0.1, format="%.2f")

    prediksi_button = st.form_submit_button("🔍 Prediksi Status")

if prediksi_button:
    input_df = pd.DataFrame([inputs])
    input_df = input_df.applymap(extract_int)

    for col in numerical_pca_1 + numerical_pca_2:
        input_df[[col]] = scalers[col].transform(input_df[[col]])

    for col in categorical_columns:
        input_df[[col]] = input_df[[col]].astype(str)
        input_df[[col]] = encoders[col].transform(input_df[[col]])

    pc1 = pca_1.transform(input_df[numerical_pca_1])
    pc2 = pca_2.transform(input_df[numerical_pca_2])

    pc1_df = pd.DataFrame(pc1, columns=[f"pc1_{i+1}" for i in range(pc1.shape[1])])
    pc2_df = pd.DataFrame(pc2, columns=[f"pc2_{i+1}" for i in range(pc2.shape[1])])

    final_df = input_df[categorical_columns].copy()
    final_df = pd.concat([final_df, pc1_df, pc2_df], axis=1)

    try:
        final_df = final_df[model.feature_names_in_]
    except:
        expected_cols = categorical_columns + [f"pc1_{i+1}" for i in range(5)] + [f"pc2_{i+1}" for i in range(2)]
        final_df = final_df[expected_cols]

    prediction = model.predict(final_df)
    label = target_encoder.inverse_transform(prediction)[0]

    st.success(f"🎯 Status Mahasiswa Diprediksi: **{label}**")
