import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle
import warnings
warnings.filterwarnings('ignore')

# 1. Load the dataset
df = pd.read_csv(r"D:\Coding\Web-Development\Loan_prediction\loan_prediction.csv")

# 2. Handle Missing Values (matching your notebook logic)
categorical_columns = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Credit_History', 'Loan_Amount_Term']
numerical_columns = ['ApplicantIncome', 'CoapplicantIncome', 'LoanAmount']

for column in categorical_columns:
    df[column].fillna(df[column].mode()[0], inplace=True)

for column in numerical_columns:
    df[column].fillna(df[column].mean(), inplace=True)

# 3. Drop unnecessary columns
df = df.drop('Loan_ID', axis=1)

# 4. Encode Categorical variables
label_encoders = {}
cat_cols_to_encode = ['Gender', 'Married', 'Dependents', 'Education', 'Self_Employed', 'Property_Area', 'Loan_Status']

for column in cat_cols_to_encode:
    le = LabelEncoder()
    df[column] = le.fit_transform(df[column].astype(str))
    label_encoders[column] = le

# 5. Split Data
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# 6. Train the Model (Using Random Forest based on your notebook's findings)
rf_classifier = RandomForestClassifier(n_estimators=1000, random_state=1, max_leaf_nodes=20)
rf_classifier.fit(X_train, y_train)

# 7. Save the model and encoders for the Streamlit app
with open('loan_model.pkl', 'wb') as f:
    pickle.dump({'model': rf_classifier, 'encoders': label_encoders}, f)

print(f"Model trained successfully. Accuracy on test set: {rf_classifier.score(X_test, y_test):.2%}")
print("Saved as 'loan_model.pkl'")