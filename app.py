import streamlit as st
import pandas as pd
import numpy as np
from lightgbm import Booster
from joblib import load
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model
model = load(r"C:\Users\kavya\Downloads\lightgbm_model.sav")

# Define the numerical columns
numerical_cols = [
    "Age",
    "Flight Distance",
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Inflight service",
    "Cleanliness",
    "Departure Delay in Minutes",
    "Arrival Delay in Minutes",
]

# Streamlit App
st.title("Flight Customer Satisfaction")

# Sidebar for user input
st.sidebar.header("User Input Features")


# Function to preprocess input data and make predictions
def predict_satisfaction(user_input):
    # Ensure the order of features matches the model's expectations
    input_df = pd.DataFrame([user_input], columns=numerical_cols)

    # Make predictions using the loaded model
    prediction = model.predict(input_df.values.reshape(1, -1))  # Reshape to 2D array
    return prediction[0]  # Extract the scalar value from the array


# Get user input
user_input = {}
user_input["Age"] = st.sidebar.slider(f"Select {'Age'}", 1, 100, 18, key="Age")
user_input["Flight Distance"] = st.sidebar.slider(
    f"Select {'Flight Distance'}", 10, 5000, 100, key="Flight Distance"
)
user_input["Inflight wifi service"] = st.sidebar.slider(
    f"Select {'Inflight wifi service'}", 0, 5, 1, key="Inflight wifi service"
)
user_input["Departure/Arrival time convenient"] = st.sidebar.slider(
    f"Select {'Departure/Arrival time convenient'}",
    0,
    5,
    1,
    key="Departure/Arrival time convenient",
)
user_input["Ease of Online booking"] = st.sidebar.slider(
    f"Select {'Ease of Online booking'}", 0, 5, 1, key="Ease of Online booking"
)
user_input["Food and drink"] = st.sidebar.slider(
    f"Select {'Food and drink'}", 0, 5, 1, key="Food and drink"
)
user_input["Online boarding"] = st.sidebar.slider(
    f"Select {'Online boarding'}", 0, 5, 1, key="Online boarding"
)
user_input["Seat comfort"] = st.sidebar.slider(
    f"Select {'Seat comfort'}", 1, 5, 1, key="Seat comfort"
)
user_input["Inflight entertainment"] = st.sidebar.slider(
    f"Select {'Inflight entertainment'}", 1, 5, 1, key="Inflight entertainment"
)
user_input["On-board service"] = st.sidebar.slider(
    f"Select {'On-board service'}", 1, 5, 1, key="On-board service"
)
user_input["Leg room service"] = st.sidebar.slider(
    f"Select {'Leg room service'}", 0, 5, 1, key="Leg room service"
)
user_input["Baggage handling"] = st.sidebar.slider(
    f"Select {'Baggage handling'}", 1, 5, 1, key="Baggage handling"
)
user_input["Checkin service"] = st.sidebar.slider(
    f"Select {'Checkin service'}", 1, 5, 1, key="Checkin service"
)
user_input["Inflight service"] = st.sidebar.slider(
    f"Select {'Inflight service'}", 1, 5, 1, key="Inflight service"
)
user_input["Cleanliness"] = st.sidebar.slider(
    f"Select {'Cleanliness'}", 1, 5, 1, key="Cleanliness"
)
user_input["Departure Delay in Minutes"] = st.sidebar.slider(
    f"Select {'Departure Delay in Minutes'}",
    0.0,
    1500.0,
    10.0,
    key="Departure Delay in Minutes",
)
user_input["Arrival Delay in Minutes"] = st.sidebar.slider(
    f"Select {'Arrival Delay in Minutes'}",
    0.0,
    1500.0,
    10.0,
    key="Arrival Delay in Minutes",
)

st.write("User Input:")
input_table = pd.DataFrame([user_input])
st.table(input_table)

if st.sidebar.button("Predict"):
    # Predict the result
    prediction = predict_satisfaction(user_input)

    # Display the prediction
    st.subheader("Prediction")
    if prediction == 0:
        st.write("The prediction is that the customer is satisfied")
    else:
        st.write("The prediction is that the customer is not satisfied")
