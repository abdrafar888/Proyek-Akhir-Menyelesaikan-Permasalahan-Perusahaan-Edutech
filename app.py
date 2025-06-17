import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Prediksi Status Mahasiswa", layout="wide")
st.title("🎓 Prediksi Status Mahasiswa")
st.markdown("Isi data berikut untuk memprediksi status mahasiswa berdasarkan informasi akademik dan demografis.")

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

def extract_int(value):
    try:
        return int(str(value).split(" - ")[0])
    except:
        return value

with st.form("form_prediksi"):
    col1, col2 = st.columns(2)
    inputs = {}

    with col1:
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
        inputs["Previous_qualification"] = st.selectbox("Kualifikasi Sebelumnya", mothers_qualification_options)
        inputs["Displaced"] = st.selectbox("Displaced", ["0 - Tidak", "1 - Ya"])
        inputs["Debtor"] = st.selectbox("Peminjam", ["0 - Tidak", "1 - Ya"])
        inputs["Tuition_fees_up_to_date"] = st.selectbox("Biaya Lunas", ["0 - Tidak", "1 - Ya"])
        inputs["Gender"] = st.selectbox("Jenis Kelamin", ["0 - Perempuan", "1 - Laki-laki"])
        inputs["Scholarship_holder"] = st.selectbox("Beasiswa", ["0 - Tidak", "1 - Ya"])

    with col2:
        for col in numerical_pca_1 + numerical_pca_2:
            label = col.replace("_", " ").capitalize()
            inputs[col] = st.number_input(label, step=0.1, format="%.2f")

    submitted = st.form_submit_button("🔍 Prediksi Status Mahasiswa")

if submitted:
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

    pred = model.predict(final_df)
    pred_label = target_encoder.inverse_transform(pred)[0]

    st.success(f"🎯 Status Mahasiswa Diprediksi: **{pred_label}**")
