import streamlit as st
import joblib
import pandas as pd


st.title("Prediction of Car Mileage")

st.markdown("Displacement, carbon emmisions, fuel type, and vehicle class fields below are being used to predict the miles per gallon (MPG)")

Displacement = st.slider("Enter Displacement", 0.6, 8.4, 0.6)

Carbon = st.slider("Enter Carbon Emissions", 22, 979, 22)

F_type = st.selectbox("Select Fuel Type", ("Gasoline", "Gasoline/Electricity", "Diesel", "Ethanol/Gas", "Ethanol", "CNG/Gasoline", "CNG"))

V_class = st.selectbox("Select Vehicle Class", ("small car", "small SUV", "midsize car", "standard SUV", "large car", "pickup", "station wagon", "special purpose", "minivan", "van"))

if st.button("Submit"):

    pk = joblib.load("pk.pkl")

    New = pd.DataFrame([[Displacement, Carbon, F_type, V_class]], 
                     columns = ["Displ", "Comb_CO2", "Fuel", "Veh_class"])
    New = New.replace(["Gasoline", "Gasoline/Electricity", "Diesel", "Ethanol/Gas", "Ethanol", "CNG/Gasoline", "CNG"], [6, 5, 4, 3, 2, 1, 0])
    New = New.replace(["small car", "small SUV", "midsize car", "standard SUV", "large car", "pickup", "station wagon", "special purpose", "minivan", "van"], [15, 14, 13, 12, 11, 10, 9, 8, 7])
    prediction = pk.predict(New)[0]
    st.text(f"This mileage is {prediction}")