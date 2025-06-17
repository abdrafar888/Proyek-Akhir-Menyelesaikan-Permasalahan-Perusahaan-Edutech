import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# SETUP
# ===============================
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide")
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("Isi form di bawah ini untuk memprediksi apakah mahasiswa berisiko dropout atau tidak.")

# ===============================
# LOAD MODEL DAN SCALER
# ===============================
model = joblib.load("models/logreg.pkl")
scaler = joblib.load("models/scaler.pkl")
data = pd.read_csv("data_bersih.csv")
fitur = data.drop(columns="Status").columns.tolist()

# ===============================
# INPUT FORM
# ===============================
with st.form("form_input"):
    st.markdown("### 📝 Formulir Data Mahasiswa")
    col1, col2 = st.columns(2)
    input_data = {}

    for i, kolom in enumerate(fitur):
        kolom_label = kolom.replace("_", " ").capitalize()

        if "nilai" in kolom.lower() or "gdp" in kolom.lower() or "%" in kolom or "rata" in kolom.lower():
            nilai = st.number_input(kolom_label, format="%.2f", step=0.01, key=kolom)
        elif data[kolom].nunique() <= 5:
            opsi = sorted(data[kolom].unique().tolist())
            nilai = st.selectbox(kolom_label, opsi, key=kolom)
        else:
            nilai = st.number_input(kolom_label, step=1, key=kolom)

        if i % 2 == 0:
            with col1:
                input_data[kolom] = nilai
        else:
            with col2:
                input_data[kolom] = nilai

    submit = st.form_submit_button("🔍 Prediksi")

# ===============================
# PREDIKSI
# ===============================
if submit:
    try:
        input_df = pd.DataFrame([input_data])[fitur]
        input_scaled = scaler.transform(input_df)
        hasil = model.predict(input_scaled)[0]
        probas = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

        if hasil == 1:
            st.error("❌ Mahasiswa diprediksi BERISIKO DROPOUT.")
        else:
            st.success("✅ Mahasiswa diprediksi TIDAK dropout.")

        if probas is not None:
            st.info(f"📊 Probabilitas Dropout: **{probas[1]*100:.2f}%**")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
