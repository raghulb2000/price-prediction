import numpy as np
import streamlit as st
import pickle

# Load model
model = pickle.load(open("car_price_model.pkl", "rb"))

# Page setup
st.set_page_config(
    page_title="Used Car Price Prediction",
    page_icon="🚗",
    layout="centered"
)

# CSS styling
st.markdown("""
<style>
.main {
    background-color: #f5f7fa;
}
.title {
    text-align: center;
    color: #1f4e79;
    font-size: 42px;
    font-weight: bold;
}
.subtitle {
    text-align: center;
    color: #555;
    font-size: 18px;
    margin-bottom: 30px;
}
.card {
    background-color: white;
    padding: 25px;
    border-radius: 18px;
    box-shadow: 0px 4px 15px rgba(0,0,0,0.1);
}
.result-box {
    background-color: #e8f5e9;
    padding: 25px;
    border-radius: 18px;
    text-align: center;
    margin-top: 25px;
    border: 2px solid #2e7d32;
}
.result-title {
    color: #2e7d32;
    font-size: 24px;
    font-weight: bold;
}
.result-price {
    color: #1b5e20;
    font-size: 38px;
    font-weight: bold;
}
.footer {
    text-align: center;
    color: #777;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("<div class='title'>🚗 Used Car Price Prediction</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='subtitle'>Predict the selling price of a used car using Machine Learning</div>",
    unsafe_allow_html=True
)

st.markdown("<div class='card'>", unsafe_allow_html=True)

st.subheader("Enter Car Details")

col1, col2 = st.columns(2)

with col1:
    present_price = st.number_input("Present Price (Lakhs)", min_value=0.0, step=0.1)
    kms_driven = st.number_input("Kilometers Driven", min_value=0, step=1000)
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col2:
    car_age = st.number_input("Car Age", min_value=0, step=1)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    seller_type = st.selectbox("Seller Type", ["Dealer", "Individual"])
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

st.markdown("</div>", unsafe_allow_html=True)

# Prediction button
if st.button("Predict Selling Price"):
    if fuel_type == "Petrol":
        fuel_petrol = 1
        fuel_diesel = 0
    elif fuel_type == "Diesel":
        fuel_petrol = 0
        fuel_diesel = 1
    else:
        fuel_petrol = 0
        fuel_diesel = 0

    seller_individual = 1 if seller_type == "Individual" else 0
    transmission_manual = 1 if transmission == "Manual" else 0

    input_features = np.array([[
        present_price,
        kms_driven,
        owner,
        car_age,
        fuel_diesel,
        fuel_petrol,
        seller_individual,
        transmission_manual
    ]])

    prediction = model.predict(input_features)

    st.markdown(f"""
    <div class='result-box'>
        <div class='result-title'>Estimated Selling Price</div>
        <div class='result-price'>{prediction[0]:.2f} Lakhs</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown(
    "<div class='footer'>Developed using Python, Streamlit and Random Forest Regressor</div>",
    unsafe_allow_html=True
)