import pandas as pd
import numpy as np
import streamlit as st
import joblib
import datetime as dt
import calendar

# Load model and encoders 
model = joblib.load('RFgridsearch_model')
ohe = joblib.load('OnehotEncoder (1)')
scaler = joblib.load('numeric_scaler')

st.title("Mobiticket Passenger Prediction")
st.write('Enter trip details to predict number of tickets:')

# --- User Input ---
travel_from = st.selectbox(
    "Departure Location", 
    ["Awendo","Homa Bay","Kehancha","Kendu Bay","Keroka","Keumbu","Kijauri","Kisii","Mbita","Migori","Ndhiwa","Nyachenge","Oyugis","Rodi","Rongo","Sirare","Sori"]
)

travel_to = st.selectbox("Arrival Location", ['Nairobi'])
car_type = st.selectbox("Boarding Vehicle", ['Bus','Shuttle'])

# --- Set max_capacity automatically based on vehicle type ---
if car_type == 'Bus':
    max_capacity = 49
else:  # Shuttle
    max_capacity = 11

st.write(f"Max Capacity for selected vehicle ({car_type}): {max_capacity}")

# --- Hours & Minutes ---
col1, col2 = st.columns(2)
with col1:
    hours = st.number_input("Hour (24hr format)", min_value=0, max_value=23, value=12, step=1)
with col2:
    minutes = st.number_input("Minutes", min_value=0, max_value=59, value=0, step=1)
st.write(f"Selected Time: {hours:02d}:{minutes:02d}")

# --- Date Picker ---
travel_date = st.date_input("Select Travel Date", dt.date(2017, 1, 1))
day_of_week = travel_date.weekday() + 1  # Monday=1, Sunday=7
month = travel_date.month
year = travel_date.year
day_name = travel_date.strftime("%A")       # Monday, Tuesday, etc.
month_name = travel_date.strftime("%B")    # January, February, etc
st.write(f"Date Selected: {travel_date} (Day: {day_name}, Month: {month_name}, Year: {year})")

is_holiday_checkbox = st.checkbox("Is this a holiday?")
is_holiday = 'Yes' if is_holiday_checkbox else 'No'

# --- Preprocessing Function ---
def preprocess(travel_from, travel_to, car_type, max_capacity, 
               day_of_week, month, year, hour, minute, is_holiday):
    """
    Prepares a single input row for prediction using the same 
    steps as the training pipeline.
    """

    # 1. Encode Categorical Features
    cat_df = pd.DataFrame([[travel_from, travel_to, car_type, is_holiday]],
                           columns=["travel_from", "travel_to", "car_type", "is_holiday"])
    cat_encoded = ohe.transform(cat_df)

    # 2. Scale Numerical Features
    num_scaled = scaler.transform([[max_capacity, year]])

    # 3. Cyclical Encoding for Time Features
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    time_features = np.array([month_sin, month_cos, dow_sin, dow_cos, hour_sin, hour_cos, minute_sin, minute_cos])

    # 4. Combine All Features (Order must match training)
    final_features = np.concatenate([cat_encoded[0], num_scaled[0], time_features]).reshape(1, -1)

    return final_features

# --- Prediction ---
if st.button("Predict Tickets"):
    X_input = preprocess(travel_from, travel_to, car_type, max_capacity, 
                         day_of_week, month, year, hours, minutes, is_holiday)
    prediction = model.predict(X_input)
    st.success(f"Predicted Tickets: {int(round(prediction[0]))}")
