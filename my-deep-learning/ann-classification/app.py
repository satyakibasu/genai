import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import pickle

# Load the trained model
model = tf.keras.models.load_model('ann_churn_model.h5')

# Load the scaler and encoder
with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('le_gender.pkl', 'rb') as f:
    encoder_gender = pickle.load(f)

with open('ohe_geography.pkl', 'rb') as f:
    encoder_geography = pickle.load(f)


## Create the Streamlit app
st.title("Customer Churn Prediction")

# Input fields
geography = st.selectbox("Geography", options= encoder_geography.categories_[0])
gender = st.selectbox("Gender", options = encoder_gender.classes_)
age = st.number_input("Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Balance", min_value=0.0, value=1000.0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)       
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=50000.0)
tenure = st.slider("Tenure", min_value=0, max_value=10, value=3)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card", options=[0, 1])
is_active_member = st.selectbox("Is Active Member", options=[0, 1])

''' 
# Example input data format: This is displayed on Streamlit app for user reference
# The input data should be provided in the following format:
input_data = {
    'CreditScore': 600,
    'Geography': 'France',          
    'Gender': 'Male',
    'Age': 40,
    'Tenure': 3,
    'Balance': 60000,
    'NumOfProducts': 2,
    'HasCrCard': 1,
    'IsActiveMember': 1,
    'EstimatedSalary': 50000
}
'''

# Prepare the input data. The input data needs to be in the same sequence as was used in the fit model. See the example above
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],  
    'HasCrCard': [has_cr_card], 
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})


# This is for Gender
input_data['Gender'] = encoder_gender.transform(input_data['Gender'])

# One -hot encode 'Geography' column
geo_encoded = encoder_geography.transform(input_data[['Geography']])   
geo_encoded_df = pd.DataFrame(geo_encoded, columns=encoder_geography.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)


# Scale the input data
input_data_scaled = scaler.transform(input_data)

# Make prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_data_scaled)
    churn_probability = prediction[0][0]
    if churn_probability > 0.5:
        st.error(f"The customer is likely to churn with a probability of {churn_probability:.2f}")
    else:
        st.success(f"The customer is unlikely to churn with a probability of {1 - churn_probability:.2f}")
