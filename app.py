import streamlit as st
import pandas as pd
import joblib

# --- Load Model, Scaler, and Features ---
model = joblib.load("Logistic Regression - Tuned - Ukulele by Yousician.pkl")
scaler = joblib.load("scaler_Ukulele by Yousician.pkl")
feature_cols = joblib.load("feature_columns_Ukulele by Yousician.pkl")

# --- Title ---
st.title("🎵 Yousician Churn Prediction App")
st.markdown("Predict whether a user will churn based on their behavioral log from Yousician Ukulele learning app.")

# --- Mode Selection ---
mode = st.radio("Select Input Mode", ["📝 Manual Input", "📁 Upload CSV"])

# ===================================
# 📝 1. MANUAL INPUT
# ===================================
if mode == "📝 Manual Input":
    st.subheader("Enter User Log Information")

    user_input = {}
    for col in feature_cols:
        user_input[col] = st.number_input(f"{col}", step=1.0 if "log" in col or "ratio" in col else 1)

    input_df = pd.DataFrame([user_input])
    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]
    pred_label = "❌ Churn" if prediction == 1 else "✅ Not Churn"

    st.markdown(f"### 🎯 Prediction Result: **{pred_label}**")

# ===================================
# 📁 2. CSV UPLOAD
# ===================================
else:
    st.subheader("Upload Preprocessed CSV File")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)

        if all(col in data.columns for col in feature_cols):
            X = data[feature_cols]
            X_scaled = scaler.transform(X)
            preds = model.predict(X_scaled)

            data["prediction"] = ["❌ Churn" if p == 1 else "✅ Not Churn" for p in preds]

            st.success("✅ Prediction completed!")
            st.dataframe(data[["prediction"] + feature_cols].head())

            # 📥 Downloadable CSV
            csv = data.to_csv(index=False).encode("utf-8")
            st.download_button("⬇️ Download Predictions", data=csv, file_name="churn_predictions.csv", mime="text/csv")
        else:
            st.error("❗CSV file must contain all required feature columns used during training.")

# Footer
st.markdown("---")
st.markdown("Built with ❤️ by Hijir Della Wirasti | Tuned Logistic Regression Model")
