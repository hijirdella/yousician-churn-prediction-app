# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import joblib

# === Load Komponen Model ===
model = joblib.load('Logistic Regression - Tuned - Ukulele by Yousician.pkl')
scaler = joblib.load('scaler_Ukulele by Yousician.pkl')
selected_features = joblib.load('feature_columns_Ukulele by Yousician.pkl')

# === Judul Aplikasi ===
st.set_page_config(page_title="Yousician Churn Prediction App", page_icon="🎸")
st.title("🎸 Yousician Churn Prediction App")

# === Deskripsi Singkat ===
st.markdown("""
Prediksi apakah pengguna akan churn (berhenti menggunakan aplikasi) berdasarkan perilaku bermain mereka dalam aplikasi *Ukulele by Yousician*.

🔍 Anda dapat memasukkan data secara manual atau mengunggah file CSV yang sudah melalui preprocessing.

📌 Model yang digunakan: **Logistic Regression (Hyperparameter Tuned)**
""")

# === Pilih Mode Input ===
mode = st.radio("📥 Pilih Mode Input", ["📝 Manual", "📁 Upload CSV"])

# === MODE 1: Input Manual ===
if mode == "📝 Manual":
    st.subheader("🔧 Input Data Manual")

    user_input = {}
    for feature in selected_features:
        if feature in ["session_index", "notes_evaluated", "notes_successful", "chords_evaluated", "chords_successful"]:
            user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0, step=1)
        else:
            user_input[feature] = st.number_input(f"{feature.replace('_', ' ').title()}", min_value=0.0)

    if st.button("🔮 Prediksi Churn"):
        input_df = pd.DataFrame([user_input])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]
        st.success(f"📈 Prediksi: {'Churn' if prediction == 1 else 'Not Churn'}")

# === MODE 2: Upload CSV ===
elif mode == "📁 Upload CSV":
    st.subheader("📂 Upload File CSV")

    st.markdown("""
    ⚠️ **Catatan:** File CSV Anda harus sudah berisi kolom-kolom fitur hasil preprocessing, yaitu:
    - `days_since_signup`, `difficulty_level`, `session_index`, `time_playing`
    - `notes_evaluated`, `notes_successful`, `chords_evaluated`, `chords_successful`, dll
    """)

    uploaded_file = st.file_uploader("Unggah file CSV preprocessed", type=["csv"])

    if uploaded_file:
        df = pd.read_csv(uploaded_file)

        missing = [col for col in selected_features if col not in df.columns]
        if missing:
            st.error(f"❌ File Anda tidak memiliki kolom: {missing}")
        else:
            st.success("✅ File berhasil dibaca dan siap diprediksi")
            df_scaled = scaler.transform(df[selected_features])
            df['churn_prediction'] = model.predict(df_scaled)

            st.write("📊 Hasil Prediksi Churn:")
            st.dataframe(df[['user_id', 'churn_prediction']] if 'user_id' in df.columns else df)

            csv_download = df.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Hasil Prediksi", data=csv_download, file_name="churn_prediction_result.csv")
