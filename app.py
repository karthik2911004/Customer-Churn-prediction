import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load model
model = joblib.load("model.pkl")

st.title("Customer Churn Prediction")

option = st.radio("Select Prediction Mode", ["Single Entry", "CSV Upload"])

# Label encoding for categorical features
def encode_inputs(df):
    mappings = {
        'gender': {'Male': 1, 'Female': 0},
        'Partner': {'Yes': 1, 'No': 0},
        'Dependents': {'Yes': 1, 'No': 0},
        'PhoneService': {'Yes': 1, 'No': 0},
        'MultipleLines': {'Yes': 1, 'No': 0, 'No phone service': 2},
        'InternetService': {'DSL': 0, 'Fiber optic': 1, 'No': 2},
        'OnlineSecurity': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'OnlineBackup': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'DeviceProtection': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'TechSupport': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'StreamingTV': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'StreamingMovies': {'Yes': 1, 'No': 0, 'No internet service': 2},
        'Contract': {'Month-to-month': 0, 'One year': 1, 'Two year': 2},
        'PaperlessBilling': {'Yes': 1, 'No': 0},
        'PaymentMethod': {
            'Electronic check': 0,
            'Mailed check': 1,
            'Bank transfer (automatic)': 2,
            'Credit card (automatic)': 3
        }
    }

    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = df[col].map(mapping)

    return df

if option == "Single Entry":
    st.markdown("### Enter Customer Details")

    # Collect all 19 features
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Senior Citizen", [0, 1])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", [
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ])
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0)
    total_charges = st.number_input("Total Charges", min_value=0.0)

    if st.button("Predict Churn"):
        # Encode input manually
        input_data = pd.DataFrame([{
            'gender': gender,
            'SeniorCitizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'tenure': tenure,
            'PhoneService': phone_service,
            'MultipleLines': multiple_lines,
            'InternetService': internet_service,
            'OnlineSecurity': online_security,
            'OnlineBackup': online_backup,
            'DeviceProtection': device_protection,
            'TechSupport': tech_support,
            'StreamingTV': streaming_tv,
            'StreamingMovies': streaming_movies,
            'Contract': contract,
            'PaperlessBilling': paperless_billing,
            'PaymentMethod': payment_method,
            'MonthlyCharges': monthly_charges,
            'TotalCharges': total_charges
        }])

        input_encoded = encode_inputs(input_data)

        try:
            prediction = model.predict(input_encoded)[0]
            if prediction == 1:
                st.error("The customer is likely to Churn.")
            else:
                st.success("The customer is likely to Stay.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

else:
    st.markdown("### Upload CSV File")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)

        required_columns = [
            'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure',
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges'
        ]

        if not all(col in df.columns for col in required_columns):
            st.error("CSV must contain the following columns:\n" + ", ".join(required_columns))
        else:
            df_encoded = encode_inputs(df[required_columns])

            try:
                predictions = model.predict(df_encoded)
                df['Prediction'] = ['Churn' if p == 1 else 'No Churn' for p in predictions]

                st.success("Prediction Completed!")
                st.dataframe(df)

                csv = df.to_csv(index=False).encode('utf-8')
                st.download_button("Download Predictions as CSV", csv, "churn_predictions.csv", "text/csv")
            except Exception as e:
                st.error(f"Error during prediction: {e}")
