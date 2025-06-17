import streamlit as st
import pandas as pd
import numpy as np
import joblib

model = joblib.load("models/logreg.pkl")
scaler = joblib.load("models/scaler.pkl")

data = pd.read_csv("data_bersih.csv")
fitur = data.drop("Status", axis=1).columns.tolist()

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("Silakan isi data mahasiswa di bawah ini untuk memprediksi apakah mereka berisiko dropout:")

input_dict = {}

mapping_dict = {
    "Gender": {"Perempuan": 0, "Laki-laki": 1},
    "Scholarship_holder": {"Tidak menerima beasiswa": 0, "Menerima beasiswa": 1},
    "Debtor": {"Bukan peminjam": 0, "Peminjam": 1},
    "Tuition_fees_up_to_date": {"Belum lunas": 0, "Sudah lunas": 1},
    "Displaced": {"Tidak": 0, "Ya": 1},
    "Educational_special_needs": {"Tidak": 0, "Ya": 1},
    "International": {"Bukan mahasiswa internasional": 0, "Mahasiswa internasional": 1}
}

for kolom in fitur:
    if kolom in mapping_dict:
        display_options = list(mapping_dict[kolom].keys())
        pilihan = st.selectbox(kolom.replace("_", " "), display_options)
        input_dict[kolom] = mapping_dict[kolom][pilihan]
    elif "nilai" in kolom.lower() or "gdp" in kolom.lower() or "%" in kolom or "rata" in kolom.lower():
        input_dict[kolom] = st.number_input(kolom.replace("_", " "), format="%.2f", step=0.01)
    elif data[kolom].nunique() <= 5:
        unique_vals = sorted(data[kolom].unique().tolist())
        input_dict[kolom] = st.selectbox(kolom.replace("_", " "), unique_vals)
    else:
        input_dict[kolom] = st.number_input(kolom.replace("_", " "), step=1)

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
        st.error("Terjadi error saat prediksi.")
