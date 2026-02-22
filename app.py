import streamlit as st
import pandas as pd
import pickle

# Set page configuration
st.set_page_config(page_title="Bank Loan Predictor", layout="centered")

# Load the trained model and encoders
@st.cache_resource
def load_model():
    with open('loan_model.pkl', 'rb') as f:
        data = pickle.load(f)
    return data['model'], data['encoders']

try:
    model, encoders = load_model()
except FileNotFoundError:
    st.error("Model file not found! Please run 'train_model.py' first to generate 'loan_model.pkl'.")
    st.stop()

st.title("🏦 Bank Loan Prediction App")
st.write("Enter the customer's details below to predict whether their loan will be approved.")

# Create a form for user inputs
with st.form("loan_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        married = st.selectbox("Married", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
        education = st.selectbox("Education", ["Graduate", "Not Graduate"])
        self_employed = st.selectbox("Self Employed", ["Yes", "No"])
        property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
        
    with col2:
        applicant_income = st.number_input("Applicant Income", min_value=0)
        coapplicant_income = st.number_input("Coapplicant Income", min_value=0.0)
        loan_amount = st.number_input("Loan Amount", min_value=0.0)
        loan_amount_term = st.number_input("Loan Amount Term (Days)", value=360.0)
        credit_history = st.selectbox("Credit History", [1.0, 0.0], format_func=lambda x: "Good (1.0)" if x == 1.0 else "Bad (0.0)")

    submit_button = st.form_submit_button(label="Predict Loan Status")

if submit_button:
    # 1. Structure the input data into a dataframe
    input_data = pd.DataFrame([[
        gender, married, dependents, education, self_employed,
        applicant_income, coapplicant_income, loan_amount,
        loan_amount_term, credit_history, property_area
    ]], columns=['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 
                 'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
                 'Loan_Amount_Term', 'Credit_History', 'Property_Area'])
    
    # 2. Encode categorical variables just like during training
    for col in ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area']:
        input_data[col] = encoders[col].transform(input_data[col])
        
    # 3. Make Prediction
    prediction = model.predict(input_data)
    result = encoders['Loan_Status'].inverse_transform(prediction)
    
    # 4. Display the result
    st.markdown("---")
    if result[0] == 'Y':
        st.success("🎉 Congratulations! The loan is predicted to be **APPROVED**.")
    else:
        st.error("🚫 Sorry. The loan is predicted to be **REJECTED**.")