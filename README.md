# 🏦 Bank Loan Prediction App

## 📌 Project Overview
The Bank Loan Prediction App is a machine learning web application built to automate the loan eligibility process. By analyzing a customer's demographic and financial details (such as income, education, credit history, and loan amount), the application predicts whether their bank loan will be **Approved** or **Rejected**. 

This project demonstrates practical end-to-end machine learning skills, from data preprocessing and feature encoding to training a classification model and deploying it as an interactive web interface.

## 🚀 Features
* **Interactive User Interface:** A clean, easy-to-use frontend built with Streamlit.
* **Automated Data Preprocessing:** Handles missing values (mean for numerical, mode for categorical) and encodes categorical features seamlessly.
* **Machine Learning Model:** Powered by a robust `RandomForestClassifier` optimized for accurate predictions.
* **Instant Predictions:** Users get immediate visual feedback on their loan status upon submitting the form.

## 🛠️ Tech Stack
* **Language:** Python
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **Data Visualization:** Matplotlib, Seaborn
* **Web Deployment:** Streamlit

## 📂 Project Structure
* `train_model.py`: The core script that loads the dataset, cleans the data, trains the Random Forest model, and serializes it.
* `app.py`: The Streamlit frontend that takes user inputs, loads the trained model, and displays the prediction.
* `loan_prediction.csv`: The historical dataset used to train and test the model.
* `loan_model.pkl`: The saved (pickled) machine learning model and label encoders.
* `requirements.txt`: A list of all Python dependencies required to run the project.

## 💻 How to Run Locally

**1. Clone the repository**
```bash
https://github.com/Ravindarnaik90/bank-loan-predictioner.git
