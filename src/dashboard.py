import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

st.title("AI Sales Forecasting Dashboard")

# Load dataset
df = pd.read_csv("train.csv")

# Convert date
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# Load model
model = joblib.load("sales_model.pkl")

# -----------------------------
# Sales Trend Chart
# -----------------------------
st.subheader("Sales Trend Over Time")

sales_trend = df.groupby("Order Date")["Sales"].sum()

fig, ax = plt.subplots()
ax.plot(sales_trend.index, sales_trend.values)
ax.set_xlabel("Date")
ax.set_ylabel("Sales")
ax.set_title("Daily Sales Trend")

st.pyplot(fig)

# -----------------------------
# Top Categories
# -----------------------------
st.subheader("Top Product Categories")

category_sales = df.groupby("Category")["Sales"].sum().sort_values(ascending=False)

st.bar_chart(category_sales)

# -----------------------------
# AI Prediction
# -----------------------------
st.subheader("AI Sales Prediction")

month = st.slider("Month", 1, 12, 6)
day = st.slider("Day", 1, 31, 15)
weekday = st.slider("Weekday (0 = Monday)", 0, 6, 2)

if st.button("Predict Sales"):
    
    input_data = pd.DataFrame({
        "month": [month],
        "day": [day],
        "weekday": [weekday]
    })

    prediction = model.predict(input_data)

    st.success(f"Predicted Sales: {prediction[0]:.2f}")


    st.subheader("7-Day AI Sales Forecast")

future_days = []

for i in range(1,8):
    future_days.append({
        "month": month,
        "day": min(day+i, 28),
        "weekday": (weekday+i)%7
    })

future_df = pd.DataFrame(future_days)

future_pred = model.predict(future_df)

forecast_df = pd.DataFrame({
    "Day":[f"Day {i}" for i in range(1,8)],
    "Predicted Sales":future_pred
})

st.line_chart(forecast_df.set_index("Day"))