import streamlit as st
import pandas as pd
import joblib

# Load model, scaler, encoder
model = joblib.load("gboost_model.joblib")
scaler = joblib.load("scaler.joblib")
encoder = joblib.load("encoder.joblib")
pca = joblib.load("pca.joblib")

st.title("Prediksi Risiko Dropout Mahasiswa")

inputs = {}

inputs["Marital_status"] = st.selectbox("Status Pernikahan", [
    "1 - Single", "2 - Married", "3 - Widower", "4 - Divorced", "5 - Facto Union", "6 - Legally Separated"
])

inputs["Application_mode"] = st.selectbox("Mode Aplikasi", [
    "1 - 1st phase - general contingent", "2 - Ordinance No. 612/93",
    "5 - 1st phase - special contingent (Azores Island)", "7 - Other higher courses",
    "10 - Ordinance No. 854-B/99", "15 - International student", "16 - Madeira contingent",
    "17 - 2nd phase", "18 - 3rd phase", "26 - Diff Plan", "27 - Other Institution",
    "39 - Over 23", "42 - Transfer", "43 - Change of course", "44 - Tech diploma holders",
    "51 - Change institution", "53 - Short cycle diploma", "57 - Intl change"
])

inputs["Course"] = st.selectbox("Program Studi", [
    "33 - Biofuel Tech", "171 - Animation", "8014 - Soc Service (evening)", "9003 - Agronomy",
    "9070 - Comm Design", "9085 - Vet Nursing", "9119 - Info Eng", "9130 - Equinculture",
    "9147 - Management", "9238 - Social Service", "9254 - Tourism", "9500 - Nursing",
    "9556 - Oral Hygiene", "9670 - Ad & Marketing", "9773 - Journalism", "9853 - Basic Ed",
    "9991 - Management (evening)"
])

mothers_qualification_options = [
    "1 - Basic 1st Cycle", "2 - Basic 2nd Cycle", "3 - Basic 3rd Cycle",
    "4 - Secondary", "5 - Higher Education - Bachelor", "6 - Higher Education - Master",
    "9 - Unknown", "10 - Not Applicable", "14 - Higher Education - Doctorate"
]

fathers_qualification_options = mothers_qualification_options

occupation_options = [
    "0 - Unemployed", "1 - Armed Forces", "2 - Management", "3 - Professionals",
    "4 - Technicians", "5 - Clerical", "6 - Service and sales", "7 - Agriculture",
    "8 - Skilled manual", "9 - Elementary", "10 - Unknown", "11 - Not Applicable"
]

inputs["Mothers_qualification"] = st.selectbox("Kualifikasi Ibu", mothers_qualification_options)
inputs["Fathers_qualification"] = st.selectbox("Kualifikasi Ayah", fathers_qualification_options)
inputs["Mothers_occupation"] = st.selectbox("Pekerjaan Ibu", occupation_options)
inputs["Fathers_occupation"] = st.selectbox("Pekerjaan Ayah", occupation_options)
inputs["Previous_qualification"] = st.selectbox("Kualifikasi Sebelumnya", mothers_qualification_options)

inputs["Displaced"] = st.selectbox("Displaced", ["0 - Tidak", "1 - Ya"])
inputs["Debtor"] = st.selectbox("Peminjam", ["0 - Tidak", "1 - Ya"])
inputs["Tuition_fees_up_to_date"] = st.selectbox("Biaya Lunas", ["0 - Tidak", "1 - Ya"])
inputs["Gender"] = st.selectbox("Jenis Kelamin", ["0 - Perempuan", "1 - Laki-laki"])
inputs["Scholarship_holder"] = st.selectbox("Beasiswa", ["0 - Tidak", "1 - Ya"])

def extract_int(value):
    return int(str(value).split(" - ")[0]) if " - " in str(value) else int(value)

if st.button("Prediksi Dropout"):
    input_df = pd.DataFrame([inputs])
    input_df = input_df.applymap(extract_int)

    # Pisahkan kolom kategorikal dan numerik
    categorical_cols = input_df.select_dtypes(include='int').columns.tolist()
    encoded = encoder.transform(input_df[categorical_cols])
    scaled = scaler.transform(encoded)
    reduced = pca.transform(scaled)

    pred = model.predict(reduced)[0]
    proba = model.predict_proba(reduced)[0][1]

    if pred == 1:
        st.error(f"⚠️ Mahasiswa diprediksi berisiko dropout (probabilitas: {proba:.2f})")
    else:
        st.success(f"✅ Mahasiswa diprediksi akan tetap lanjut studi (probabilitas dropout: {proba:.2f})")
