import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/logreg.pkl")
scaler = joblib.load("models/scaler.pkl")

st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="centered")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Masukkan informasi akademik mahasiswa untuk memprediksi kemungkinan dropout.")

# Pastikan kolom ini SAMA dengan yang digunakan saat training
feature_columns = [
    'Admission_grade', 'Previous_qualification_grade', 'Age_at_enrollment',
    'Curricular_units_1st_sem_enrolled', 'Curricular_units_1st_sem_evaluations',
    'Curricular_units_1st_sem_approved', 'Curricular_units_1st_sem_grade',
    'Curricular_units_2nd_sem_enrolled', 'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_2nd_sem_approved', 'Curricular_units_2nd_sem_grade',
    'Unemployment_rate', 'Inflation_rate', 'GDP'
]

inputs = {}
with st.form("prediction_form"):
    for col in feature_columns:
        label = col.replace("_", " ").capitalize()
        inputs[col] = st.number_input(label, step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
    input_df = pd.DataFrame([inputs])

    # Jaga-jaga agar urutan dan nama kolom 100% cocok
    input_df = input_df[feature_columns]

    # Transformasi scaling
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)[0]
    label = "Dropout" if prediction == 1 else "Tidak Dropout"

    st.success(f"📊 Hasil Prediksi: **{label}**")
    st.write("Kolom input:", input_df.columns.tolist())
    st.write("Kolom yang diminta scaler:", scaler.feature_names_in_)


