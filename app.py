import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide", page_icon="🎓")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Prototype untuk mendeteksi mahasiswa berisiko dropout berdasarkan data akademik dan demografis.")

data = pd.read_csv("data_bersih.csv")
models = {
    "Logistic Regression": joblib.load("models/logreg.pkl"),
    "Random Forest": joblib.load("models/randomforest.pkl"),
    "Decision Tree": joblib.load("models/decisiontree.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "Naive Bayes": joblib.load("models/naivebayes.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
}
scaler = joblib.load("models/scaler.pkl")

st.sidebar.header("🧠 Pilih Model")
model_name = st.sidebar.selectbox("Model", list(models.keys()))
menu = st.sidebar.radio("Pilih Mode", ["Evaluasi Model", "Prediksi Mahasiswa Baru"])

if menu == "Evaluasi Model":
    if st.sidebar.button("🚀 Jalankan Model"):
        X = data.drop(columns=["Status"])
        y = data["Status"]
        X_scaled = scaler.transform(X)
        model = models[model_name]
        y_pred = model.predict(X_scaled)
        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, target_names=["Not Dropout", "Dropout"])

        st.subheader(f"📊 Evaluasi Model: {model_name}")
        st.markdown(f"**🎯 Akurasi:** `{acc:.4f}`")

        st.markdown("#### 📌 Confusion Matrix")
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Dropout", "Dropout"])
        disp.plot(ax=ax, cmap="Blues", values_format="d")
        st.pyplot(fig)

        st.markdown("#### 🧾 Classification Report")
        st.text(report)

        st.markdown("#### 🔍 Data Mahasiswa")
        st.dataframe(X, use_container_width=True)

elif menu == "Prediksi Mahasiswa Baru":
    st.subheader("📝 Formulir Data Mahasiswa Baru")

    with st.form("form_prediksi"):
        input_data = {}
        col1, col2 = st.columns(2)

        with col1:
            input_data["Admission_grade"] = st.number_input("Admission Grade", min_value=0.0, max_value=200.0, step=0.1)
            input_data["Age_at_enrollment"] = st.number_input("Umur Saat Mendaftar", min_value=15, max_value=100)
            input_data["Unemployment_rate"] = st.number_input("Tingkat Pengangguran", step=0.1)
            input_data["Inflation_rate"] = st.number_input("Tingkat Inflasi", step=0.1)
            input_data["GDP"] = st.number_input("GDP", step=0.1)
            input_data["Scholarship_holder"] = st.selectbox("Penerima Beasiswa", [0, 1])
            input_data["Tuition_fees_up_to_date"] = st.selectbox("Biaya Kuliah Lunas", [0, 1])

        with col2:
            input_data["Curricular_units_1st_sem_grade"] = st.number_input("Nilai Semester 1", step=0.1)
            input_data["Curricular_units_2nd_sem_grade"] = st.number_input("Nilai Semester 2", step=0.1)
            input_data["Curricular_units_1st_sem_approved"] = st.number_input("Lulus Semester 1", step=1)
            input_data["Curricular_units_2nd_sem_approved"] = st.number_input("Lulus Semester 2", step=1)
            input_data["Curricular_units_1st_sem_enrolled"] = st.number_input("Ambil Semester 1", step=1)
            input_data["Curricular_units_2nd_sem_enrolled"] = st.number_input("Ambil Semester 2", step=1)
            input_data["Gender"] = st.selectbox("Jenis Kelamin", [0, 1])

        submitted = st.form_submit_button("🔍 Prediksi Dropout")

    if submitted:
        df_input = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df_input)
        model = models[model_name]
        pred = model.predict(df_scaled)
        label = "Dropout" if pred[0] == 1 else "Tidak Dropout"

        st.success(f"🎯 Prediksi Status Mahasiswa: **{label}**")

st.markdown("---")
st.markdown("© 2025 Abdul Rafar · Jaya Jaya Institut")
