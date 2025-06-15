import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, ConfusionMatrixDisplay
import joblib

# =============================
# Pengaturan Halaman
# =============================
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide", page_icon="🎓")

st.markdown("""
    <style>
    .main {background-color: #0e1117; color: white;}
    </style>
""", unsafe_allow_html=True)

# =============================
# Judul Aplikasi
# =============================
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Prototype untuk mendeteksi mahasiswa berisiko dropout berdasarkan data akademik dan demografis.")

# =============================
# Load Model dan Data
# =============================
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

# =============================
# Sidebar: Pilih Model
# =============================
st.sidebar.header("🧠 Pilih Model")
model_name = st.sidebar.selectbox("Model", list(models.keys()))
run_button = st.sidebar.button("🚀 Jalankan Model")

# =============================
# Jalankan Model dan Evaluasi
# =============================
if run_button:
    # Persiapkan data
    X = data.drop(columns=["Status"])
    y = data["Status"]
    X_scaled = scaler.transform(X)

    # Prediksi
    model = models[model_name]
    y_pred = model.predict(X_scaled)

    # Evaluasi
    acc = accuracy_score(y, y_pred)
    cm = confusion_matrix(y, y_pred)
    report = classification_report(y, y_pred, target_names=["Not Dropout", "Dropout"])

    # Tampilkan hasil evaluasi
    st.subheader(f"📊 Hasil Evaluasi Model: {model_name}")
    st.markdown(f"**🎯 Akurasi:** `{acc:.4f}`")

    st.markdown("#### 📌 Confusion Matrix")
    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Dropout", "Dropout"])
    disp.plot(ax=ax, cmap="Blues", values_format="d")
    st.pyplot(fig)

    st.markdown("#### 🧾 Classification Report")
    st.text(report)

    # Tampilkan data mahasiswa di bagian bawah
    st.markdown("#### 🔍 Data Mahasiswa")
    st.dataframe(X, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("© 2025 Abdul Rafar · Jaya Jaya Institut")
