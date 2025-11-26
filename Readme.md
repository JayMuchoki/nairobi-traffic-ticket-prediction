[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_red.svg)](https://nairobi-traffic-ticket-prediction.streamlit.app/)

# 🚦 **Traffic Movement into Nairobi – Passenger Prediction App**

<img width="2328" height="1091" alt="image" src="https://github.com/user-attachments/assets/9cde0dae-e0d7-4ac5-bfad-219d00c8aff6" />


## 📌 Project Overview

Traffic congestion in Nairobi is heavily influenced by the movement of vehicles entering the city, especially from the **Western Region**. Factors such as:

* Time passengers board vehicles,
* Types of vehicles used,
* Number of rides from specific locations,
* Travel dates and holidays,

all contribute to traffic buildup.

This project analyzes these patterns and builds a predictive model to forecast the **expected passenger count per ride**, helping optimize transportation planning.

---

## 🎯 Problem Statement

Transport operators often dispatch vehicles (buses, shuttles, etc.) without knowing:

* Whether the vehicle will be fully occupied,
* The best time to schedule rides,
* Passenger traffic patterns during peak congestion hours.

This leads to:

* Empty seats (revenue loss),
* Poor planning of ride schedules,
* Increased congestion due to poorly timed entries into Nairobi.

This project solves that by predicting **how many passengers to expect per ride**.

---

## 🧠 Objectives

* Predict the number of passengers expected for each ride.
* Identify the optimal vehicle type (bus vs shuttle) based on demand.
* Understand the best departure times for maximizing occupancy.
* Support transport planning that reduces congestion during peak hours.

---

## 🧩 Dataset & Features Used

The dataset contained:

* **Vehicle Type** (Bus, Shuttle, etc.)
* **Region of Travel** (from Western regions)
* **Travel Date & Time**
* **Holiday Indicator** (whether travel date was a holiday)

### Feature Engineering

* Encoding categorical variables (vehicle type, region, holiday).
* Handling numerical features (time of day, date-based patterns).
* Handling numerical features
* Scaling numerical variables using StandardScaler to normalize values and improve model performance


---

## 🤖 Machine Learning Models Used

We tested multiple algorithms:

* **Linear Regression** → R² = **0.38** (data was not linear)
* **XGBoost** → Moderate performance
* **Random Forest** → **Best Model** with R² = **0.61**

A score of 0.61 means:

> The model explains **61% of the variation** in passenger bookings.

With more detailed features, the model could capture additional hidden patterns and improve further.

---

## 🌐 Streamlit Application

A Streamlit app was built to make real-time predictions.

### The app accepts:

* Departure location
* Vehicle type
* Travel date
* Travel time

### The app outputs:

* **Predicted number of passengers** for that ride

### How it helps

* Operators can choose the right vehicle (bus vs shuttle)
* Optimize schedules to avoid traffic-heavy hours
* Improve vehicle occupancy
* Reduce unnecessary congestion entering Nairobi

---

## 🛠️ Tech Stack

* Python
* Pandas / NumPy
* Scikit-learn
* XGBoost
* Streamlit
* Matplotlib / Seaborn (for visualizations)

---

## 🚀 How to Run the Streamlit App

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## 📦 Project Structure

```
├── data/
├── notebooks/
├── app.py
├── models/
├── README.md
└── requirements.txt
```

---

## 📝 Conclusion

Understanding passenger flow patterns is essential for reducing congestion in Nairobi. By combining data analysis and machine learning, this project provides insights and predictions that enable smarter transportation planning.

The Streamlit app acts as a practical tool for transport operators to make data-driven decisions in real time.


---

## 🔗 Live Demo

Access the deployed Streamlit application here:
👉 **[https://nairobi-traffic-ticket-prediction.streamlit.app/](https://nairobi-traffic-ticket-prediction.streamlit.app/)**

---

## 👤 Author

Jay Muchoki


