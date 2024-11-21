import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained RandomForest model (replace 'random_forest_model.pkl' with your actual model path)
with open('random_forest_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up Streamlit app title
st.title(" 🌾  Crop Yield Prediction  🌾")

# Display 4 crop images in a row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image("maize.jpeg", caption="Maize", width=150)
with col2:
    st.image("potatoes.jpeg", caption="Potatoes", width=150)
with col3:
    st.image("rice.jpeg", caption="Rice", width=150)
with col4:
    st.image("sorghum.jpeg", caption="Sorghum", width=150)

# Input fields for prediction
st.header("Input Features")

# Year, Rainfall, Pesticides, Average Temperature as number inputs
year = st.number_input("Year", min_value=1900, max_value=2100, step=1)
rainfall = st.number_input("Average Rainfall (mm/year)", min_value=0.0, step=1.0)
pesticides = st.number_input("Pesticides Used (tonnes)", min_value=0.0, step=1.0)
avg_temp = st.number_input("Average Temperature (°C)", min_value=-50.0, max_value=50.0, step=0.1)

# List of possible countries (or areas) and crops (items)
areas = ['Albania', 'Algeria', 'Angola', 'Argentina', 'Armenia', 'Australia', 'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh', 'Belarus', 'Belgium', 'Botswana', 'Brazil', 'Bulgaria', 'Burkina Faso', 'Burundi', 'Cameroon', 'Canada', 'Central African Republic', 'Chile', 'Colombia', 'Croatia', 'Denmark', 'Dominican Republic', 'Ecuador', 'Egypt', 'El Salvador', 'Eritrea', 'Estonia', 'Finland', 'France', 'Germany', 'Ghana', 'Greece', 'Guatemala', 'Guinea', 'Guyana', 'Haiti', 'Honduras', 'Hungary', 'India', 'Indonesia', 'Iraq', 'Ireland', 'Italy', 'Jamaica', 'Japan', 'Kazakhstan', 'Kenya', 'Latvia', 'Lebanon', 'Lesotho', 'Libya', 'Lithuania', 'Madagascar', 'Malawi', 'Malaysia', 'Mali', 'Mauritania', 'Mauritius', 'Mexico', 'Montenegro', 'Morocco', 'Mozambique', 'Namibia', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua', 'Niger', 'Norway', 'Pakistan', 'Papua New Guinea', 'Peru', 'Poland', 'Portugal', 'Qatar', 'Romania', 'Rwanda', 'Saudi Arabia', 'Senegal', 'Slovenia', 'South Africa', 'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Sweden', 'Switzerland', 'Tajikistan', 'Thailand', 'Tunisia', 'Turkey', 'Uganda', 'Ukraine', 'United Kingdom', 'Uruguay', 'Zambia', 'Zimbabwe']
items = ['Maize', 'Potatoes', 'Rice, paddy', 'Sorghum', 'Soybeans', 'Wheat', 'Cassava', 'Sweet potatoes', 'Plantains and others', 'Yams']

area = st.selectbox("Area", areas)
item = st.selectbox("Crop Type", items)

# Prepare the input data for prediction
input_data = {
    'Area': [area],
    'Item': [item],
    'Year': [year],
    'average_rain_fall_mm_per_year': [rainfall],
    'pesticides_tonnes': [pesticides],
    'avg_temp': [avg_temp]
}

# Convert to DataFrame
input_df = pd.DataFrame(input_data)

# **Recreate LabelEncoder** and encode the categorical columns (Area and Item)
label_en_area = LabelEncoder()
label_en_item = LabelEncoder()

label_en_area.fit(areas)  # Fit LabelEncoder on the area values
label_en_item.fit(items)  # Fit LabelEncoder on the item values

# Transform the input values
input_df['Area'] = label_en_area.transform(input_df['Area'])
input_df['Item'] = label_en_item.transform(input_df['Item'])

# When the user clicks the 'Predict Crop Yield' button
if st.button("Predict Crop Yield in Hectares"):
    # Prediction
    predicted_yield = model.predict(input_df)

    # Display the result
    st.success(f"Predicted Crop Yield: {predicted_yield[0]:.2f} hg/ha")
