import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load your trained model
model = joblib.load('churn_model.pkl')

# Define the Streamlit app
def app():
    st.title("Customer Churn Prediction")

    # Categorical variables

    state = st.selectbox('Select State', ['AK', 'AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', 'HI', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY'])
    state_map = { 'AK': 0, 'AL': 1, 'AR': 2, 'AZ': 3, 'CA': 4, 
        'CO': 5, 'CT': 6, 'DC': 7, 'DE': 8, 'FL': 9, 
        'GA': 10, 'HI': 11, 'IA': 12, 'ID': 13, 'IL': 14, 
        'IN': 15, 'KS': 16, 'KY': 17, 'LA': 18, 'MA': 19, 
        'MD': 20, 'ME': 21, 'MI': 22, 'MN': 23, 'MO': 24, 
        'MS': 25, 'MT': 26, 'NC': 27, 'ND': 28, 'NE': 29, 
        'NH': 30, 'NJ': 31, 'NM': 32, 'NV': 33, 'NY': 34, 
        'OH': 35, 'OK': 36, 'OR': 37, 'PA': 38, 'RI': 39, 
        'SC': 40, 'SD': 41, 'TN': 42, 'TX': 43, 'UT': 44, 
        'VA': 45, 'VT': 46, 'WA': 47, 'WI': 48, 'WV': 49, 
        'WY': 50}
    state_encoded = state_map[state]

    international_plan = st.radio("Do you have an International Plan?", ('Yes', 'No'))
    international_plan_encoded = 1 if international_plan == 'Yes' else 0

    voicemail_plan = st.radio("Do you have a Voice Plan?", ('Yes', 'No'))
    voicemail_plan_encoded = 1 if voicemail_plan == 'Yes' else 0

    # Numerical variables

    area_code = st.number_input("Enter your area code", min_value=0, max_value=999, step=1)
    account_length = st.number_input("Enter your account length", min_value=0)
    voicemail_messages = st.number_input("Enter number of voicemail messages", min_value=0)
    international_minutes = st.number_input("Enter international minutes", min_value=0)
    international_calls = st.number_input("Enter number of international calls", min_value=0)
    international_charge = st.number_input("Enter international charge", min_value=0.0, format="%.2f")
    day_minutes = st.number_input("Enter day minutes", min_value=0)
    day_calls = st.number_input("Enter day calls", min_value=0)
    day_charge = st.number_input("Enter day charge", min_value=0.0, format="%.2f")
    evening_minutes = st.number_input("Enter evening minutes", min_value=0)
    evening_calls = st.number_input("Enter evening calls", min_value=0)
    evening_charge = st.number_input("Enter evening charge", min_value=0.0, format="%.2f")
    night_minutes = st.number_input("Enter night minutes", min_value=0)
    night_calls = st.number_input("Enter night calls", min_value=0)
    night_charge = st.number_input("Enter night charge", min_value=0.0, format="%.2f")
    customer_service = st.number_input("Enter number of customer service calls", min_value=0)
    churn = st.selectbox("Has the customer churned?", options=["Yes", "No"])

    # Create a button to make the prediction
    if st.button("Predict"):
        # Prepare the input data for prediction (reshape into 2D array)
        input_data = np.array([[area_code, account_length,
            voicemail_messages, international_minutes,
            international_calls, international_charge, day_minutes,
            day_calls, day_charge, evening_minutes, evening_calls,
            evening_charge, night_minutes, night_calls, night_charge,
            customer_service, state_encoded, international_plan_encoded, voicemail_plan_encoded]])

        # Get the prediction from the model
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)

        # Display the result
        if prediction == 0:
            st.write("The customer is **Loyal**.")
        else:
            st.write("The customer is **Churned**.")

        st.write(f"Prediction Probability: {prediction_proba[0][1]:.2f}")

if __name__ == "__main__":
    app()
