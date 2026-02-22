import streamlit as st
import pandas as pd
import pickle

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Bank Loan Predictor", layout="centered", page_icon="🏦")

# --- CUSTOM CSS FOR 3D ANIMATED UI ---
page_bg_css = """
<style>
/* 1. Animated Gradient Background */
.stApp {
    background: linear-gradient(-45deg, #1a1a2e, #16213e, #0f3460, #e94560);
    background-size: 400% 400%;
    animation: gradientBG 15s ease infinite;
}

@keyframes gradientBG {
    0% { background-position: 0% 50%; }
    50% { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}

/* 2. 3D Glassmorphism Floating Card for the Form */
[data-testid="stForm"] {
    background: rgba(255, 255, 255, 0.05);
    border-radius: 20px;
    padding: 30px;
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.2);
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
    border: 1px solid rgba(255, 255, 255, 0.1);
    transform: perspective(1000px) translateZ(0px) rotateX(0deg);
    transition: transform 0.4s ease-out, box-shadow 0.4s ease-out;
}

[data-testid="stForm"]:hover {
    transform: perspective(1000px) translateZ(20px) rotateX(2deg);
    box-shadow: 0 20px 40px rgba(0, 0, 0, 0.6), inset 0 1px 0 rgba(255, 255, 255, 0.3);
}

/* 3. 3D Push Button Effect */
div.stButton > button {
    background: linear-gradient(to right, #ff416c, #ff4b2b);
    color: white !important;
    font-weight: bold;
    border: none;
    border-radius: 12px;
    padding: 10px 24px;
    box-shadow: 0 6px #8b0000; /* Dark red bottom shadow for 3D effect */
    transition: all 0.1s ease;
    width: 100%;
}

div.stButton > button:active {
    box-shadow: 0 2px #8b0000;
    transform: translateY(4px); /* Pushes the button down */
}

div.stButton > button:hover {
    filter: brightness(1.2);
}

/* Make text readable on dark background */
h1, h2, h3, p, label {
    color: white !important;
    text-shadow: 1px 1px 2px rgba(0,0,0,0.5);
}
</style>
"""
st.markdown(page_bg_css, unsafe_allow_html=True)
# --------------------------------------

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

    submit_button = st.form_submit_button(label="🚀 Predict Loan Status")

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
        st.balloons() # Adds a nice built-in Streamlit animation
    else:
        st.error("🚫 Sorry. The loan is predicted to be **REJECTED**.")
