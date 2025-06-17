import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# LOAD MODEL & SCALER
# ===============================
model = joblib.load("models/logreg.pkl")  # ganti sesuai model yang ingin dipakai
scaler = joblib.load("models/scaler.pkl")

# ===============================
# UI CONFIGURATION
# ===============================
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Isi formulir berikut untuk memprediksi apakah mahasiswa berpotensi **dropout** atau tidak.")

# ===============================
# INPUT FORM
# ===============================
with st.form("form"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
        inputs["Age_at_enrollment"] = st.number_input("Usia saat mendaftar", min_value=15, max_value=100, value=18)
        inputs["Admission_grade"] = st.number_input("Nilai masuk (Admission Grade)", min_value=0.0, max_value=200.0)
        inputs["Previous_qualification_grade"] = st.number_input("Nilai kualifikasi sebelumnya", min_value=0.0, max_value=200.0)
        inputs["Unemployment_rate"] = st.number_input("Tingkat pengangguran (%)", step=0.1)
        inputs["Inflation_rate"] = st.number_input("Tingkat inflasi (%)", step=0.1)
        inputs["GDP"] = st.number_input("GDP", step=0.1)
        inputs["Scholarship_holder"] = st.selectbox("Penerima beasiswa?", ["0", "1"])
        inputs["Tuition_fees_up_to_date"] = st.selectbox("Pembayaran biaya kuliah lancar?", ["0", "1"])

    with col2:
        inputs["Curricular_units_1st_sem_enrolled"] = st.number_input("Mata kuliah semester 1 diambil", step=1)
        inputs["Curricular_units_1st_sem_approved"] = st.number_input("Mata kuliah semester 1 lulus", step=1)
        inputs["Curricular_units_1st_sem_evaluations"] = st.number_input("Evaluasi semester 1", step=1)
        inputs["Curricular_units_1st_sem_grade"] = st.number_input("Rata-rata nilai semester 1", step=0.1)
        inputs["Curricular_units_2nd_sem_enrolled"] = st.number_input("Mata kuliah semester 2 diambil", step=1)
        inputs["Curricular_units_2nd_sem_approved"] = st.number_input("Mata kuliah semester 2 lulus", step=1)
        inputs["Curricular_units_2nd_sem_evaluations"] = st.number_input("Evaluasi semester 2", step=1)
        inputs["Curricular_units_2nd_sem_grade"] = st.number_input("Rata-rata nilai semester 2", step=0.1)

    submitted = st.form_submit_button("🔍 Prediksi")

# ===============================
# PREDIKSI
# ===============================
if submitted:
    input_df = pd.DataFrame([inputs])
    input_scaled = scaler.transform(input_df.values)
    prediction = model.predict(input_scaled)[0]
    label = "Dropout" if prediction == 1 else "Tidak Dropout"

    st.success(f"🎯 Prediksi Status Mahasiswa: **{label}**")
