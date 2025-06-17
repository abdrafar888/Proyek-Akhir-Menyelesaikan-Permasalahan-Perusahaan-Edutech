import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load model dan scaler
model = joblib.load("models/logreg.pkl")  # Ganti dengan model lain jika perlu
scaler = joblib.load("models/scaler.pkl")

# Load data bersih
data = pd.read_csv("data_bersih.csv")
fitur = data.drop("Status", axis=1).columns.tolist()

st.title("🎓 Prediksi Risiko Dropout Mahasiswa")
st.markdown("Silakan isi data mahasiswa di bawah ini:")

# Opsi lengkap
mothers_qualification_options = [
    "1 - Secondary Education", "2 - Bachelor's Degree", "3 - Degree", "4 - Master's", "5 - Doctorate",
    "6 - Frequency of Higher Education", "9 - 12th Year Not Completed", "10 - 11th Year Not Completed",
    "11 - 7th Year (Old)", "12 - Other - 11th Year", "14 - 10th Year", "18 - General Commerce",
    "19 - Basic Ed 3rd Cycle", "22 - Technical-professional", "26 - 7th year",
    "27 - 2nd cycle high school", "29 - 9th Year Not Completed", "30 - 8th year", "34 - Unknown",
    "35 - Can't read/write", "36 - Can read (no 4th year)", "37 - Basic Ed 1st Cycle",
    "38 - Basic Ed 2nd Cycle", "39 - Tech specialization", "40 - Degree (1st cycle)",
    "41 - Specialized studies", "42 - Prof. higher tech", "43 - Master (2nd cycle)",
    "44 - Doctorate (3rd cycle)"
]

fathers_qualification_options = [
    "1 - Secondary Education", "2 - Bachelor's Degree", "3 - Degree", "4 - Master's", "5 - Doctorate",
    "6 - Frequency of Higher Education", "9 - 12th Year Not Completed", "10 - 11th Year Not Completed",
    "11 - 7th Year (Old)", "12 - Other - 11th Year", "13 - 2nd year complementary", "14 - 10th Year",
    "18 - General Commerce", "19 - Basic Ed 3rd Cycle", "20 - Complementary High School",
    "22 - Technical-professional", "25 - Comp. High School - not completed", "26 - 7th year",
    "27 - 2nd cycle high school", "29 - 9th Year Not Completed", "30 - 8th year", "31 - Admin & Commerce",
    "33 - Accounting & Admin", "34 - Unknown", "35 - Can't read/write", "36 - Can read (no 4th year)",
    "37 - Basic Ed 1st Cycle", "38 - Basic Ed 2nd Cycle", "39 - Tech specialization",
    "40 - Degree (1st cycle)", "41 - Specialized studies", "42 - Prof. higher tech",
    "43 - Master (2nd cycle)", "44 - Doctorate (3rd cycle)"
]

occupation_options = [
    "0 - Student", "1 - Legislative/Executive", "2 - Scientific Specialists", "3 - Technicians",
    "4 - Admin Staff", "5 - Services/Sellers", "6 - Farmers", "7 - Construction Workers",
    "8 - Machine Operators", "9 - Unskilled Workers", "10 - Armed Forces", "90 - Other", "99 - Blank",
    "101 - Armed Forces Officers", "102 - Sergeants", "103 - Other Armed Forces",
    "112 - Admin Service Directors", "114 - Hotel/Trade Directors", "121 - Science/Engineering",
    "122 - Health Professionals", "123 - Teachers", "124 - Finance Specialists", "125 - ICT Specialists",
    "131 - Mid Sci/Eng Tech", "132 - Mid Health Tech", "134 - Mid Legal/Cultural", "135 - ICT Tech",
    "141 - Secretaries/Data Ops", "143 - Finance Ops", "144 - Admin Support", "151 - Personal Service",
    "152 - Sellers", "153 - Care Workers", "154 - Security", "161 - Market Farmers",
    "163 - Subsistence Farmers", "171 - Skilled Construction", "172 - Metalworkers", "173 - Artisans",
    "174 - Electricians", "175 - Food/Wood/Clothing", "181 - Plant Operators", "182 - Assemblers",
    "183 - Drivers", "191 - Cleaners", "192 - Unskilled Agriculture", "193 - Unskilled Construction",
    "194 - Meal Prep", "195 - Street Vendors"
]

# Input manual sesuai fitur
inputs = {}
inputs["Educational_special_needs"] = st.selectbox("Kebutuhan Khusus", ["0 - Tidak", "1 - Ya"])
inputs["Debtor"] = st.selectbox("Status Pinjaman", ["0 - Tidak", "1 - Ya"])
inputs["Tuition_fees_up_to_date"] = st.selectbox("Biaya Kuliah Lunas", ["0 - Tidak", "1 - Ya"])
inputs["Gender"] = st.selectbox("Jenis Kelamin", ["0 - Perempuan", "1 - Laki-laki"])
inputs["Scholarship_holder"] = st.selectbox("Penerima Beasiswa", ["0 - Tidak", "1 - Ya"])
inputs["Age_at_enrollment"] = st.number_input("Usia Saat Mendaftar", step=1)
inputs["International"] = st.selectbox("Mahasiswa Internasional", ["0 - Tidak", "1 - Ya"])
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
inputs["Mothers_qualification"] = st.selectbox("Kualifikasi Ibu", mothers_qualification_options)
inputs["Fathers_qualification"] = st.selectbox("Kualifikasi Ayah", fathers_qualification_options)
inputs["Mothers_occupation"] = st.selectbox("Pekerjaan Ibu", occupation_options)
inputs["Fathers_occupation"] = st.selectbox("Pekerjaan Ayah", occupation_options)
inputs["Previous_qualification"] = st.selectbox("Kualifikasi Sebelumnya", mothers_qualification_options)  # asumsi sama dgn ibu

# Tambah input numerik lainnya dari data jika ada
for kolom in fitur:
    if kolom not in inputs:
        inputs[kolom] = st.number_input(kolom, step=1)

# Proses prediksi
if st.button("🔍 Prediksi Dropout"):
    try:
        df = pd.DataFrame([inputs])
        for col in df.columns:
            if df[col].astype(str).str.contains("-").any():
                df[col] = df[col].str.split(" - ").str[0].astype(int)
        df = df[fitur]  # pastikan urutan kolom sama
        df_scaled = scaler.transform(df)
        pred = model.predict(df_scaled)[0]
        proba = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None

        if pred == 1:
            st.error("❌ Mahasiswa diprediksi BERISIKO DROPOUT.")
        else:
            st.success("✅ Mahasiswa diprediksi TIDAK dropout.")

        if proba is not None:
            st.info(f"Probabilitas Dropout: {proba * 100:.2f}%")
    except Exception as e:
        st.error(f"Terjadi kesalahan saat prediksi: {str(e)}")
