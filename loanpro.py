# Import necessary libraries
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv("D:\\resume project\\loan\\Training.csv")

# Data Cleaning: Drop rows with missing values
df = df.dropna(how="any")

# Drop 'Loan_ID' column
df = df.drop("Loan_ID", axis=1)

# Separate features and target variable
X = df.drop("Loan_Status", axis=1)
y = df["Loan_Status"]

# Separate numerical and categorical columns
numerical_cols = X.select_dtypes(include=["number"]).columns
categorical_cols = X.select_dtypes(include=["object"]).columns

# Create transformers for numerical and categorical data
numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])
categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

# Create a column transformer to apply different transformers to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_transformer, numerical_cols),
        ("cat", categorical_transformer, categorical_cols),
    ]
)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model Training
model = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", RandomForestClassifier(random_state=42))])
model.fit(X_train, y_train)

# Model Evaluation
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit App
st.title("Loan Prediction App")

# Sidebar for User Input
st.sidebar.header("User Input Features")

# Get user inputs for prediction
def get_user_inputs():
    gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
    married = st.sidebar.selectbox("Marital Status", ["Yes", "No"])
    dependents = st.sidebar.selectbox("Number of Dependents", ["0", "1", "2", "3"])
    education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
    self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])
    applicant_income = st.sidebar.slider("Applicant Income", 150, 81000, 2500)
    coapplicant_income = st.sidebar.slider("Coapplicant Income", 0, 41667, 0)

    # Use a selectbox for 'Loan Amount' with available values
    loan_amount_options = [
        128.,  66., 120., 141., 267.,  95., 158., 168., 349.,  70., 200.,
       114.,  17., 125.,  76., 133., 104., 315., 116., 191., 122., 110.,
        35.,  74., 106., 320., 144., 184.,  80.,  47., 134.,  44., 100.,
       112., 286.,  97.,  96., 135., 180.,  99., 165., 258., 126., 312.,
       136., 172.,  81., 187., 113., 176., 111., 167.,  50., 210., 175.,
       131., 188.,  25., 137., 115., 151., 225., 216.,  94., 185., 154.,
       259., 194., 160., 102., 290.,  84.,  88., 242., 129.,  30., 118.,
       152., 244., 600., 255.,  98., 275., 121.,  75.,  63.,  87., 101.,
       495.,  73., 260., 108.,  48., 164., 170.,  83.,  90., 166., 124.,
        55.,  59., 127., 214., 240., 130.,  60., 280., 140., 155., 123.,
       201., 138., 279., 192., 304., 150., 207., 436.,  78.,  54.,  89.,
       139.,  93., 132., 480.,  56., 300., 376.,  67., 117.,  71., 173.,
        46., 228., 308., 105., 236., 570., 380., 296., 156., 109., 103.,
        45.,  65.,  53., 360.,  62., 218., 178., 239., 143., 148., 149.,
       153., 162., 230.,  86., 234., 246., 500., 119., 107., 209., 208.,
       243.,  40., 250., 311., 400., 161., 324., 157., 145., 181.,  26.,
       182., 211.,   9., 186., 205.,  36., 146., 142., 496., 253

        # Add the rest of the values
    ]
    loan_amount = st.sidebar.selectbox("Loan Amount", loan_amount_options)

    loan_term = st.sidebar.selectbox("Loan Term (Months)", ["360", "120", "180",  "60", "300", "480", "240",  "36",  "84"])
    credit_history = st.sidebar.selectbox("Credit History", [0, 1])
    property_area = st.sidebar.selectbox("Property Area", ["Rural", "Semiurban", "Urban"])

    user_inputs = {
        "Gender": gender,
        "Married": married,
        "Dependents": dependents,
        "Education": education,
        "Self_Employed": self_employed,
        "ApplicantIncome": applicant_income,
        "CoapplicantIncome": coapplicant_income,
        "LoanAmount": loan_amount,
        "Loan_Amount_Term": loan_term,
        "Credit_History": credit_history,
        "Property_Area": property_area,
    }

    return user_inputs

user_inputs = get_user_inputs()

# Display user inputs
st.write("### User Inputs:")
st.write(user_inputs)

# Preprocess user inputs for prediction
# (You may need to perform similar preprocessing as done during training)

# Make prediction
prediction = model.predict(pd.DataFrame([user_inputs]))

# Display prediction
st.write("### Prediction:")
st.write("Loan Status:", prediction[0])

# Model Evaluation Metrics
st.write("### Model Evaluation:")
st.write("Accuracy:", accuracy)
st.write("Classification Report:")
st.write(classification_report(y_test, y_pred))
st.write("Confusion Matrix:")
st.write(confusion_matrix(y_test, y_pred))
