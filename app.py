import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler

# Cache the model loading
@st.cache_resource
def load_model():
    return pickle.load(open('random_forest_model.pkl', 'rb'))

# Cache the scaler initialization
@st.cache_resource
def load_scaler():
    return StandardScaler()

# Load model and scaler once
rf_model = load_model()
scaler = load_scaler()

# Title of the web app
st.title("Customer Churn Prediction")

# Frontend input fields (Updated)
SeniorCitizen = st.selectbox("Is the customer a senior citizen?", options=['Yes', 'No'])
Partner = st.selectbox("Does the customer have a partner?", options=['Yes', 'No'])
Dependents = st.selectbox("Does the customer have any dependents?", options=['Yes', 'No'])
tenure = st.number_input("Enter tenure (how long the customer has been with the company) (in months)", min_value=0)
OnlineSecurity = st.selectbox("Does the customer have online security?", options=['Yes', 'No', 'No internet service'])
OnlineBackup = st.selectbox("Does the customer have online backup?", options=['Yes', 'No', 'No internet service'])
DeviceProtection = st.selectbox("Does the customer have device protection?", options=['Yes', 'No', 'No internet service'])
TechSupport = st.selectbox("Does the customer have tech support?", options=['Yes', 'No', 'No internet service'])
Contract = st.selectbox("Enter the contract type of the customer", options=['Month-to-month', 'One year', 'Two years'])
PaperlessBilling = st.selectbox("Is the billing paperless?", options=['Yes', 'No'])
PaymentMethod = st.selectbox("Select the payment method", options=['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
MonthlyCharges = st.number_input("Enter the monthly charges of the customer", min_value=0.0)
TotalCharges = st.number_input("Enter the total charges of the customer", min_value=0.0)

# Helper function to preprocess and predict
def predictive_rf(SeniorCitizen, Partner, Dependents, tenure, 
                  OnlineSecurity, OnlineBackup, DeviceProtection, 
                  TechSupport, Contract, PaperlessBilling, 
                  PaymentMethod, MonthlyCharges, TotalCharges):
    data = {
        'SeniorCitizen': [1 if SeniorCitizen == 'Yes' else 0],  # Yes=1, No=0
        'Partner': [1 if Partner == 'Yes' else 0],  # Yes=1, No=0
        'Dependents': [1 if Dependents == 'Yes' else 0],  # Yes=1, No=0
        'tenure': [tenure],  # Numerical
        'OnlineSecurity': [1 if OnlineSecurity == 'Yes' else (0 if OnlineSecurity == 'No' else -1)],  # Yes=1, No=0, No internet service=-1
        'OnlineBackup': [1 if OnlineBackup == 'Yes' else (0 if OnlineBackup == 'No' else -1)],  # Yes=1, No=0, No internet service=-1
        'DeviceProtection': [1 if DeviceProtection == 'Yes' else (0 if DeviceProtection == 'No' else -1)],  # Yes=1, No=0, No internet service=-1
        'TechSupport': [1 if TechSupport == 'Yes' else (0 if TechSupport == 'No' else -1)],  # Yes=1, No=0, No internet service=-1
        'Contract': [0 if Contract == 'Month-to-month' else (1 if Contract == 'One year' else 2)],  # Month-to-month=0, One year=1, Two year=2
        'PaperlessBilling': [1 if PaperlessBilling == 'Yes' else 0],  # Yes=1, No=0
        'PaymentMethod': [0 if PaymentMethod == 'Electronic check' else 
                         (1 if PaymentMethod == 'Mailed check' else 
                          (2 if PaymentMethod == 'Bank transfer (automatic)' else 
                           (3 if PaymentMethod == 'Credit card (automatic)' else 4)))],  # 0-4 based on method
        'MonthlyCharges': [MonthlyCharges],  # Numerical
        'TotalCharges': [TotalCharges]  # Numerical
    }

    df_input = pd.DataFrame(data)

    # Scale numerical features
    df_input[['tenure', 'TotalCharges', 'MonthlyCharges']] = scaler.fit_transform(df_input[['tenure', 'TotalCharges', 'MonthlyCharges']])

    # Predict using the Random Forest model
    result = rf_model.predict(df_input)
    return result[0]

# Button to trigger prediction
if st.button("Predict"):
    result = predictive_rf(SeniorCitizen, Partner, Dependents, tenure, 
                           OnlineSecurity, OnlineBackup, DeviceProtection, 
                           TechSupport, Contract, PaperlessBilling, 
                           PaymentMethod, MonthlyCharges, TotalCharges)
    if result == 0:
        st.success("The customer is **not likely** to churn.")
    else:
        st.warning("The customer is **likely** to churn.")
