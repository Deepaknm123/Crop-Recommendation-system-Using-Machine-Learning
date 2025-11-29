import streamlit as st
import pickle
import numpy as np

# Load the trained model
# with open('crop_recommendation_model.pkl', 'rb') as file:
#     model = pickle.load(file)

# Dropdown for model selection
# model_choice = st.selectbox("Choose the model you want to use:", ["Random Forest", "SVM", "XGBoost"])

# Load selected model
# if model_choice == "Random Forest":
#     model_path = 'crop_recommendation_model.pkl'
# elif model_choice == 'XGBoost':
#     model_path = 'xgb_model.pkl'
# else: model_path = 'svm_model.pkl'

model_path = 'svm_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)


# Set page configuration
st.set_page_config(page_title="Crop Recommendation System", page_icon="🌾", layout="centered")

# Set background color
st.markdown(
    """
    <style>
    .stApp {
        # background-color: #0FE09D;
    }
    .input-field {
        font-size: 1.5rem;
        color: white;
    }
    .prediction-box {
        border-radius: 10px;
        padding: 10px;
        background-color: white;
        color: black;
        margin-top: 10px;
        
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the app
st.markdown("<h1 style='text-align: center;'>Crop Recommendation System🌱</h1>", unsafe_allow_html=True)

# Input fields
st.markdown("### Enter the following details:")
col1, col2, col3 = st.columns(3)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, format="%.2f", key="N", help="Enter the Nitrogen content in the soil")
with col2:
    P = st.number_input("Phosphorus (P)", min_value=0.0, format="%.2f", key="P", help="Enter the Phosphorus content in the soil")
with col3:
    K = st.number_input("Potassium (K)", min_value=0.0, format="%.2f", key="K", help="Enter the Potassium content in the soil")

col4, col5, col6 = st.columns(3)

with col4:
    temperature = st.number_input("Temperature (°C)", min_value=0.0, format="%.2f", key="temperature", help="Enter the temperature in Celsius")
with col5:
    humidity = st.number_input("Humidity (%)", min_value=0.0, format="%.2f", key="humidity", help="Enter the humidity percentage")
with col6:
    ph = st.number_input("pH", min_value=0.0, format="%.2f", key="ph", help="Enter the pH value of the soil")

rainfall = st.number_input("Rainfall (mm)", min_value=0.0, format="%.2f", key="rainfall", help="Enter the rainfall in mm")

# Predict button
if st.button("Predict"):
    try:
        # Create input array for prediction
        input_data = np.array([[N, P, K, temperature, humidity, ph, rainfall]])

        # Predict the crop
        prediction = model.predict_proba(input_data)[0]
        top_3_indices = prediction.argsort()[-3:][::-1]
        top_3_crops = [model.classes_[i] for i in top_3_indices]

        # Display the recommended crops
        st.markdown("## Predicted Suitable Crops just for You⬇️:")
        for i, crop in enumerate(top_3_crops, 1):
            st.markdown(f"<div class='prediction-box'><h4>{i}) {crop}</h4></div>", unsafe_allow_html=True)
    except ValueError:
        st.error("Please enter valid numerical values.")
