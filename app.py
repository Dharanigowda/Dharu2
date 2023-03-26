import streamlit as st
import pandas as pd
import numpy as np
import pickle
import xgboost as xgb

# Load saved model from pickle file
with open('xgb_model.pkl', 'rb') as file:
    clf = pickle.load(file)

def main():
    # Set the app title
    st.title('Telecom Customer Churn Prediction')
    
    # Define input fields for prediction
    gender = st.selectbox('Gender', ['Male', 'Female'])
    seniorcitizen = st.selectbox('Senior Citizen', [0, 1])
    partner = st.selectbox('Partner', ['Yes', 'No'])
    dependents = st.selectbox('Dependents', ['Yes', 'No'])
    tenure = st.slider('Tenure (in months)', min_value=0, max_value=72, step=1)
    phoneservice = st.selectbox('Phone Service', ['Yes', 'No'])
    multiplelines = st.selectbox('Multiple Lines', ['No', 'Yes', 'No phone service'])
    internetservice = st.selectbox('Internet Service', ['DSL', 'Fiber optic', 'No'])
    onlinesecurity = st.selectbox('Online Security', ['No', 'Yes', 'No internet service'])
    onlinebackup = st.selectbox('Online Backup', ['No', 'Yes', 'No internet service'])
    deviceprotection = st.selectbox('Device Protection', ['No', 'Yes', 'No internet service'])
    techsupport = st.selectbox('Tech Support', ['No', 'Yes', 'No internet service'])
    streamingtv = st.selectbox('Streaming TV', ['No', 'Yes', 'No internet service'])
    streamingmovies = st.selectbox('Streaming Movies', ['No', 'Yes', 'No internet service'])
    contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
    paperlessbilling = st.selectbox('Paperless Billing', ['Yes', 'No'])
    paymentmethod = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
    monthlycharges = st.slider('Monthly Charges (in dollars)', min_value=0, max_value=150, step=1)
    totalcharges = st.slider('Total Charges (in dollars)', min_value=0, max_value=10000, step=1)
    
    # Prepare input data for prediction
    new_data = pd.DataFrame({
        'gender': [gender],
        'SeniorCitizen': [seniorcitizen],
        'Partner': [partner],
        'Dependents': [dependents],
        'tenure': [tenure],
        'PhoneService': [phoneservice],
        'MultipleLines': [multiplelines],
        'InternetService': [internetservice],
        'OnlineSecurity': [onlinesecurity],
        'OnlineBackup': [onlinebackup],
        'DeviceProtection': [deviceprotection],
        'TechSupport': [techsupport],
        'StreamingTV': [streamingtv],
        'StreamingMovies': [streamingmovies],
        'Contract': [contract],
        'PaperlessBilling': [paperlessbilling],
        'PaymentMethod': [paymentmethod],
        'MonthlyCharges': [monthlycharges],
        'TotalCharges': [totalcharges]
    })

    # Convert categorical columns to numeric codes
    cat_cols = new_data.select_dtypes(include=['category']).columns
    new_data[cat_cols] = new_data[cat_cols].apply(lambda x: x.cat.codes)

    # Create DMatrix for prediction
    dnew = xgb.DMatrix(new_data, enable_categorical=True)

    # Make predictions
    predictions = clf.predict(dnew)
    print(predictions)
    
if __name__=='__main__':
    main()
