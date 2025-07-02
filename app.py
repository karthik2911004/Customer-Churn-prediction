import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

st.title("Customer Churn Prediction")

option = st.radio("Select Prediction Mode", ["Single Entry", "CSV Upload"])

def encode_inputs(df):
    df['gender'] = df['gender'].map({'Male': 1, 'Female': 0})
    df['Partner'] = df['Partner'].map({'Yes': 1, 'No': 0})
    df['Dependents'] = df['Dependents'].map({'Yes': 1, 'No': 0})
    return df

if option == "Single Entry":
    st.markdown("### Enter Customer Details")

    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Has Partner?", ["Yes", "No"])
    dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
    tenure = st.number_input("Tenure (in months)", min_value=0, max_value=72)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

    if st.button("Predict Churn"):
        gender = 1 if gender == "Male" else 0
        partner = 1 if partner == "Yes" else 0
        dependents = 1 if dependents == "Yes" else 0

        features = np.array([[gender, senior, partner, dependents, tenure, monthly_charges, total_charges]])
        prediction = model.predict(features)[0]

        if prediction == 1:
            st.error("The customer is likely to Churn.")
        else:
            st.success("The customer is likely to Stay.")

else:
    st.markdown("### Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        try:
            required_columns = ['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'MonthlyCharges', 'TotalCharges']
            if not all(col in df.columns for col in required_columns):
                st.error("CSV must contain these columns: " + ", ".join(required_columns))
            else:
                df = encode_inputs(df[required_columns])
                predictions = model.predict(df)
                df['Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]

                st.success("Prediction Completed!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", csv, "churn_predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Error processing file: {e}")
