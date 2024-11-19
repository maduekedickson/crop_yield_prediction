import streamlit as st
import pickle
import pandas as pd

# Load the saved model and preprocessor
dtr = pickle.load(open('dtr.pkl', 'rb'))
preprocessor = pickle.load(open('preprocessor.pkl', 'rb'))

# App title
st.title("Crop Yield Prediction App 🌾")

# Display crop images in a row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.image("maize.jpeg", caption="Maize", width=150)
with col2:
    st.image("potatoes.jpeg", caption="Potatoes", width=150)
with col3:
    st.image("rice.jpeg", caption="Rice", width=150)
with col4:
    st.image("sorghum.jpeg", caption="Sorghum", width=150)

# Input fields
st.header("Input Features")

year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, step=1.0)
pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, step=1.0)
avg_temp = st.number_input("Average Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)

area = st.selectbox("Area", ["Albania", "Other"])  # Add all area options if available
item = st.selectbox("Crop Type", ["Maize", "Potatoes", "Rice", "Sorghum"])  # Add other crop options if needed

# Predict button
if st.button("Predict Crop Yield in Hectares"):
    # Create input DataFrame
    input_data = pd.DataFrame({
        "Year": [year],
        "average_rain_fall_mm_per_year": [rainfall],
        "pesticides_tonnes": [pesticides],
        "avg_temp": [avg_temp],
        "Area": [area],
        "Item": [item]
    })

    # Preprocess the data
    processed_data = preprocessor.transform(input_data)

    # Make prediction
    prediction = dtr.predict(processed_data)

    # Display result
    st.success(f"Predicted Crop Yield: {prediction[0]:.2f} hg/ha")
