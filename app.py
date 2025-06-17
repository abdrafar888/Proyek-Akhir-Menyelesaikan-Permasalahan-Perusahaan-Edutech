import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("models/logreg.pkl")
scaler = joblib.load("models/scaler.pkl")
data = pd.read_csv("data_bersih.csv")
fitur = data.drop("Status", axis=1).columns.tolist()

# Set Streamlit layout
st.set_page_config(page_title="Prediksi Risiko Dropout", layout="wide")
st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("Isi data mahasiswa di bawah ini untuk memprediksi apakah mereka berisiko dropout.")

with st.form("dropout_form"):
    col1, col2 = st.columns(2)
    input_dict = {}

    with col1:
        input_dict["Marital_status"] = st.selectbox("Status Pernikahan", [
            "1 - Single", "2 - Married", "3 - Widower", "4 - Divorced", "5 - Facto Union", "6 - Legally Separated"
        ])
        input_dict["Application_mode"] = st.selectbox("Mode Aplikasi", [
            "1 - 1st phase - general contingent", "2 - Ordinance No. 612/93",
            "5 - 1st phase - special contingent (Azores Island)", "7 - Other higher courses",
            "10 - Ordinance No. 854-B/99", "15 - International student", "16 - Madeira contingent",
            "17 - 2nd phase", "18 - 3rd phase", "26 - Diff Plan", "27 - Other Institution",
            "39 - Over 23", "42 - Transfer", "43 - Change of course", "44 - Tech diploma holders",
            "51 - Change institution", "53 - Short cycle diploma", "57 - Intl change"
        ])
        input_dict["Course"] = st.selectbox("Program Studi", [
            "33 - Biofuel Tech", "171 - Animation", "8014 - Soc Service (evening)", "9003 - Agronomy",
            "9070 - Comm Design", "9085 - Vet Nursing", "9119 - Info Eng", "9130 - Equinculture",
            "9147 - Management", "9238 - Social Service", "9254 - Tourism", "9500 - Nursing",
            "9556 - Oral Hygiene", "9670 - Ad & Marketing", "9773 - Journalism", "9853 - Basic Ed",
            "9991 - Management (evening)"
        ])
        input_dict["Mothers_qualification"] = st.selectbox("Kualifikasi Ibu", data["Mothers_qualification"].unique())
        input_dict["Fathers_qualification"] = st.selectbox("Kualifikasi Ayah", data["Fathers_qualification"].unique())
        input_dict["Mothers_occupation"] = st.selectbox("Pekerjaan Ibu", data["Mothers_occupation"].unique())
        input_dict["Fathers_occupation"] = st.selectbox("Pekerjaan Ayah", data["Fathers_occupation"].unique())
        input_dict["Previous_qualification"] = st.selectbox("Kualifikasi Sebelumnya", data["Previous_qualification"].unique())
        input_dict["Displaced"] = st.selectbox("Displaced", [0, 1])
        input_dict["Debtor"] = st.selectbox("Peminjam", [0, 1])
        input_dict["Tuition_fees_up_to_date"] = st.selectbox("Biaya Lunas", [0, 1])
        input_dict["Gender"] = st.selectbox("Jenis Kelamin", [0, 1])
        input_dict["Scholarship_holder"] = st.selectbox("Beasiswa", [0, 1])

    with col2:
        for kolom in fitur:
            if kolom in input_dict:  # skip kolom yang sudah diisi manual di col1
                continue
            if "nilai" in kolom.lower() or "gdp" in kolom.lower() or "%" in kolom or "rata" in kolom.lower():
                input_dict[kolom] = st.number_input(kolom.replace("_", " ").capitalize(), format="%.2f", step=0.01)
            elif data[kolom].nunique() <= 5:
                unique_vals = sorted(data[kolom].unique().tolist())
                input_dict[kolom] = st.selectbox(kolom.replace("_", " ").capitalize(), unique_vals)
            else:
                input_dict[kolom] = st.number_input(kolom.replace("_", " ").capitalize(), step=1)

    submitted = st.form_submit_button("🔍 Prediksi")

if submitted:
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
