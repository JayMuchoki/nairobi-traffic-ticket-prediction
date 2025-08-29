import pandas as pd
import numpy as np
import streamlit as st
import joblib
import datetime as dt

# ===============================
# Load model and encoders 
# ===============================
model = joblib.load('RFgridsearch_model')
ohe = joblib.load('OnehotEncoder (1)')
scaler = joblib.load('numeric_scaler')

# ===============================
# App Title & Description
# ===============================
st.set_page_config(page_title="Mobiticket Prediction", page_icon="🚌", layout="wide")

st.title("Mobiticket Passenger Prediction")
st.markdown(
    """
    This app helps **Mobiticket** forecast the number of passengers traveling 
    into **Nairobi**.  
    Use it to **predict bookings** for a single trip or in **bulk (CSV upload)**.
    """
)

# ===============================
# Preprocessing Functions
# ===============================
def preprocess(travel_from, travel_to, car_type, max_capacity, 
               day_of_week, month, year, hour, minute, is_holiday):
    """Prepare single input row for prediction."""
    cat_df = pd.DataFrame([[travel_from, travel_to, car_type, is_holiday]],
                           columns=["travel_from", "travel_to", "car_type", "is_holiday"])
    cat_encoded = ohe.transform(cat_df)

    num_scaled = scaler.transform([[max_capacity, year]])

    # Cyclical time encoding
    month_sin = np.sin(2 * np.pi * month / 12)
    month_cos = np.cos(2 * np.pi * month / 12)
    dow_sin = np.sin(2 * np.pi * day_of_week / 7)
    dow_cos = np.cos(2 * np.pi * day_of_week / 7)
    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)
    minute_sin = np.sin(2 * np.pi * minute / 60)
    minute_cos = np.cos(2 * np.pi * minute / 60)

    time_features = np.array([month_sin, month_cos, dow_sin, dow_cos,
                              hour_sin, hour_cos, minute_sin, minute_cos])

    final_features = np.concatenate([cat_encoded[0], num_scaled[0], time_features]).reshape(1, -1)
    return final_features


def preprocess_batch(df):
    df["max_capacity"] = df["car_type"].apply(lambda x: 49 if str(x).lower() == "bus" else 11)
    df["travel_date"] = pd.to_datetime(df["travel_date"])

    df["day_of_week"] = df["travel_date"].dt.weekday + 1
    df["month"] = df["travel_date"].dt.month
    df["year"] = df["travel_date"].dt.year

    cat_df = df[["travel_from", "travel_to", "car_type", "is_holiday"]]
    cat_encoded = ohe.transform(cat_df)

    num_df = df[["max_capacity", "year"]].to_numpy()
    num_scaled = scaler.transform(num_df)

    month_sin = np.sin(2 * np.pi * df["month"] / 12)
    month_cos = np.cos(2 * np.pi * df["month"] / 12)
    dow_sin = np.sin(2 * np.pi * df["day_of_week"] / 7)
    dow_cos = np.cos(2 * np.pi * df["day_of_week"] / 7)
    hour_sin = np.sin(2 * np.pi * df["hour"] / 24)
    hour_cos = np.cos(2 * np.pi * df["hour"] / 24)
    minute_sin = np.sin(2 * np.pi * df["minute"] / 60)
    minute_cos = np.cos(2 * np.pi * df["minute"] / 60)

    time_features = np.vstack([month_sin, month_cos, dow_sin, dow_cos,
                               hour_sin, hour_cos, minute_sin, minute_cos]).T

    final_features = np.hstack([cat_encoded, num_scaled, time_features])
    return final_features, df

# ===============================
# Tabs for Navigation
# ===============================
tab1, tab2 = st.tabs([" Single Trip Prediction", "📂 Batch Prediction (CSV)"])

# -------------------------------
# Single Prediction Tab
# -------------------------------
with tab1:
    st.header(" Single Trip Prediction")

    col1, col2 = st.columns(2)
    with col1:
        travel_from = st.selectbox(
            "Departure Location", 
            ["Awendo","Homa Bay","Kehancha","Kendu Bay","Keroka","Keumbu",
             "Kijauri","Kisii","Mbita","Migori","Ndhiwa","Nyachenge",
             "Oyugis","Rodi","Rongo","Sirare","Sori"]
        )
    with col2:
        travel_to = st.selectbox("Arrival Location", ['Nairobi'])

    col3, col4 = st.columns(2)
    with col3:
        car_type = st.radio("Boarding Vehicle", ['Bus','Shuttle'])
    with col4:
        max_capacity = 49 if car_type == 'Bus' else 11
        st.info(f"🚍 Max Capacity: **{max_capacity}** seats")

    col5, col6 = st.columns(2)
    with col5:
        hours = st.number_input("Hour (24hr)", min_value=0, max_value=23, value=12, step=1)
    with col6:
        minutes = st.number_input("Minutes", min_value=0, max_value=59, value=0, step=1)

    travel_date = st.date_input("Select Travel Date", dt.date(2017, 1, 1))
    day_of_week = travel_date.weekday() + 1
    month = travel_date.month
    year = travel_date.year

    is_holiday = st.checkbox("Is this a holiday?") 
    is_holiday = 'Yes' if is_holiday else 'No'

    if st.button("🚀 Predict Tickets"):
        X_input = preprocess(travel_from, travel_to, car_type, max_capacity, 
                             day_of_week, month, year, hours, minutes, is_holiday)
        prediction = model.predict(X_input)
        st.success(f"✅ Predicted Tickets: **{int(round(prediction[0]))}**")


# -------------------------------
# Batch Prediction Tab
# -------------------------------
with tab2:
    st.header("📂 Batch Prediction from CSV")

    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("📊 Preview of uploaded data:")
        st.dataframe(df.head())

        if st.button("🚀 Run Batch Prediction"):
            X_batch, df_processed = preprocess_batch(df)
            predictions = model.predict(X_batch)
            df_processed["Predicted_Tickets"] = predictions.round().astype(int)

            st.success("✅ Batch predictions complete!")
            st.dataframe(df_processed)

            csv_out = df_processed.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions", 
                               data=csv_out, 
                               file_name="predictions.csv", 
                               mime="text/csv")
