import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ---------------------- HEADER ----------------------
st.set_page_config(page_title="Prediksi Dropout Mahasiswa", layout="wide")
st.title("🎓 Prediksi Dropout Mahasiswa")
st.markdown("Prototype untuk mendeteksi mahasiswa berisiko dropout berdasarkan data akademik dan demografis.")

# ---------------------- LOAD DATA ----------------------
@st.cache_data
def load_data():
    df = pd.read_csv("data_bersih.csv")
    return df

df = load_data()

st.subheader("🔍 Data Mahasiswa")
st.dataframe(df.head())

# ---------------------- SPLIT & SCALE ----------------------
X = df.drop('Status', axis=1)
y = df['Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------- PILIH MODEL ----------------------
st.sidebar.header("🧠 Pilih Model")
model_name = st.sidebar.selectbox("Model", [
    "Logistic Regression", "Random Forest", "Decision Tree",
    "SVM", "Naive Bayes", "KNN"
])

model_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(),
    "Decision Tree": DecisionTreeClassifier(),
    "SVM": SVC(),
    "Naive Bayes": GaussianNB(),
    "KNN": KNeighborsClassifier()
}

if st.sidebar.button("🚀 Jalankan Model"):
    st.subheader(f"📊 Hasil Evaluasi Model: {model_name}")
    model = model_dict[model_name]
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    st.write(f"**🎯 Akurasi:** {acc:.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.write("**📌 Confusion Matrix:**")
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Not Dropout', 'Dropout'],
                yticklabels=['Not Dropout', 'Dropout'])
    plt.xlabel("Prediksi")
    plt.ylabel("Aktual")
    st.pyplot(fig)

    # Classification Report
    st.write("**📋 Classification Report:**")
    report = classification_report(
        y_test, y_pred, target_names=["Not Dropout", "Dropout"], output_dict=True
    )
    st.dataframe(pd.DataFrame(report).transpose())
