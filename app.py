import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("models/logreg.pkl")  # atau model lain sesuai pilihan
scaler = joblib.load("models/scaler.pkl")

# Load data bersih untuk ambil urutan dan nama kolom
data = pd.read_csv("data_bersih.csv")
fitur = data.drop("Status", axis=1).columns.tolist()

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")

st.markdown("Silakan isi data mahasiswa di bawah ini untuk memprediksi apakah mereka berisiko dropout:")

# Buat input sesuai fitur
input_dict = {}

for kolom in fitur:
    if "nilai" in kolom.lower() or "gdp" in kolom.lower() or "%" in kolom or "rata" in kolom.lower():
        # Kolom numerik pecahan
        input_dict[kolom] = st.number_input(f"{kolom}", format="%.2f", step=0.01)
    elif data[kolom].nunique() <= 5:
        # Kolom kategori (biasa bernilai 0/1 atau 0-4)
        unique_vals = sorted(data[kolom].unique().tolist())
        input_dict[kolom] = st.selectbox(f"{kolom}", unique_vals)
    else:
        # Kolom numerik integer biasa
        input_dict[kolom] = st.number_input(f"{kolom}", step=1)

# Ketika tombol ditekan
if st.button("🔍 Prediksi"):
    try:
        input_df = pd.DataFrame([input_dict])[fitur]
        input_scaled = scaler.transform(input_df)
        hasil = model.predict(input_scaled)[0]
        probas = model.predict_proba(input_scaled)[0] if hasattr(model, "predict_proba") else None

        if hasil == 1:
            st.error("❌ Mahasiswa diprediksi BERISIKO DROPOUT.")
        else:
            st.success("✅ Mahasiswa diprediksi TIDAK dropout.")

        if probas is not None:
            st.info(f"Probabilitas Dropout: {probas[1]*100:.2f}%")
    except Exception as e:
        st.error(f"Terjadi error saat prediksi: {str(e)}")
