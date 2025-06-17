import streamlit as st
import pandas as pd
import numpy as np
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
scaler = joblib.load("models/scaler.pkl")
models = {
    "Logistic Regression": joblib.load("models/logreg.pkl"),
    "Random Forest": joblib.load("models/randomforest.pkl"),
    "Decision Tree": joblib.load("models/decisiontree.pkl"),
    "SVM": joblib.load("models/svm.pkl"),
    "Naive Bayes": joblib.load("models/naivebayes.pkl"),
    "KNN": joblib.load("models/knn.pkl"),
}

# =============================
# Sidebar: Pilih Model
# =============================
st.sidebar.header("🧠 Pilih Model")
model_name = st.sidebar.selectbox("Model", list(models.keys()))

# =============================
# Tab: Evaluasi vs Prediksi Baru
# =============================
tab1, tab2 = st.tabs(["🔎 Evaluasi Model", "🧠 Prediksi Mahasiswa Baru"])

# =============================
# Tab 1: Evaluasi Model
# =============================
with tab1:
    if st.button("🚀 Jalankan Evaluasi"):
        X = data.drop(columns=["Status"])
        y = data["Status"]
        X_scaled = scaler.transform(X)
        model = models[model_name]
        y_pred = model.predict(X_scaled)

        acc = accuracy_score(y, y_pred)
        cm = confusion_matrix(y, y_pred)
        report = classification_report(y, y_pred, target_names=["Not Dropout", "Dropout"])

        st.subheader(f"📊 Hasil Evaluasi Model: {model_name}")
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

# =============================
# Tab 2: Prediksi Mahasiswa Baru
# =============================
with tab2:
    st.markdown("Masukkan data mahasiswa baru di bawah ini untuk memprediksi apakah ia berisiko dropout.")

    contoh = data.drop(columns=["Status"]).iloc[0]  # ambil kolom dan urutannya
    input_data = {}
    for col in contoh.index:
        val = st.number_input(f"{col}", value=float(contoh[col]), format="%.2f")
        input_data[col] = val

    if st.button("🔍 Prediksi Dropout"):
        df_input = pd.DataFrame([input_data])
        df_scaled = scaler.transform(df_input)
        model = models[model_name]
        pred = model.predict(df_scaled)[0]
        prob = model.predict_proba(df_scaled)[0][1] if hasattr(model, "predict_proba") else None

        st.subheader("📢 Hasil Prediksi")
        if pred == 1:
            st.error(f"🚨 Mahasiswa ini **berisiko DROP OUT**.")
        else:
            st.success(f"✅ Mahasiswa ini **TIDAK berisiko dropout**.")

        if prob is not None:
            st.markdown(f"**Probabilitas Dropout:** `{prob:.2%}`")

# =============================
# Footer
# =============================
st.markdown("---")
st.markdown("© 2025 Abdul Rafar · Jaya Jaya Institut")
