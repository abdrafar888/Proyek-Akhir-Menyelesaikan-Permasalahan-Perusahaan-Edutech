import streamlit as st
import pandas as pd
import joblib

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = joblib.load("models/logreg.pkl")   # Ganti ke knn.pkl, randomforest.pkl, dst kalau mau
scaler = joblib.load("models/scaler.pkl")

# ===============================
# FORM INPUT
# ===============================
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Masukkan informasi akademik mahasiswa untuk memprediksi kemungkinan dropout.")

# Daftar fitur numerik yang kamu punya (disesuaikan dari data aslinya)
input_cols = [
    'Admission_grade', 'Previous_qualification_grade', 'Age_at_enrollment',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

inputs = {}
with st.form("prediction_form"):
    for col in input_cols:
        label = col.replace("_", " ").capitalize()
        inputs[col] = st.number_input(label, step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔍 Prediksi")

# ===============================
# PREDIKSI
# ===============================
if submitted:
    input_df = pd.DataFrame([inputs])

    # Scaling
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    label = "Dropout" if prediction == 1 else "Tidak Dropout"

    st.success(f"📊 Hasil Prediksi: **{label}**")
