import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
import matplotlib.pyplot as plt

# ========== STEP 1: DATA LOADING ==========
@st.cache_data
def load_data():
    sales_data = pd.read_csv("prediction.csv", parse_dates=["Date"])
    sales_data["Month"] = sales_data["Date"].dt.to_period("M")
    return sales_data

sales_data = load_data()

# ========== STEP 2: SIDEBAR FILTERS ==========
st.sidebar.header("Filter Options")
product_filter = st.sidebar.multiselect(
    "Select Product Line(s)", sales_data["Product line"].unique(), default=sales_data["Product line"].unique()
)
city_filter = st.sidebar.multiselect(
    "Select City", sales_data["City"].unique(), default=sales_data["City"].unique()
)

filtered_data = sales_data[
    (sales_data["Product line"].isin(product_filter)) & (sales_data["City"].isin(city_filter))
]

# ========== STEP 3: DASHBOARD HEADER ==========
st.title("Advanced Sales Forecasting and Customer Insights Dashboard")
st.write("Analyze historical trends, forecast future sales, and understand customer behavior.")

# ========== STEP 4: HISTORICAL SALES ==========
st.header("1. Historical Sales Trends")
# Fix: Specify only the 'Total' column to sum to avoid summing datetime columns
sales_trend = filtered_data.groupby(filtered_data["Date"].dt.date)['Total'].sum().reset_index()
fig = px.line(sales_trend, x="Date", y="Total", title="Sales Over Time", labels={"Total": "Total Sales"})
st.plotly_chart(fig)

# ========== STEP 5: FORECASTING MODELS ==========
st.header("2. Forecasting Models")

# Prepare data for ARIMA
arima_data = sales_trend.set_index("Date")["Total"]
arima_model = ARIMA(arima_data, order=(1, 1, 1))
arima_result = arima_model.fit()

# Forecast with ARIMA
arima_forecast = arima_result.forecast(steps=30)
arima_forecast_df = pd.DataFrame({
    "Date": pd.date_range(start=arima_data.index[-1] + pd.Timedelta(days=1), periods=30),
    "Forecast": arima_forecast
})

# Forecast with Prophet
prophet_data = sales_trend.rename(columns={"Date": "ds", "Total": "y"})
prophet_data['ds'] = pd.to_datetime(prophet_data['ds'])  # Ensure 'ds' is datetime
prophet_model = Prophet()
prophet_model.fit(prophet_data)
future = prophet_model.make_future_dataframe(periods=30)
prophet_forecast = prophet_model.predict(future)
prophet_forecast_df = prophet_forecast[["ds", "yhat"]].rename(columns={"ds": "Date", "yhat": "Forecast"}).tail(30)

# Comparison of Models
st.subheader("Forecast Comparison")
comparison_fig = px.line(arima_forecast_df, x="Date", y="Forecast", title="ARIMA vs Prophet Forecast")
comparison_fig.add_scatter(x=prophet_forecast_df["Date"], y=prophet_forecast_df["Forecast"], mode="lines", name="Prophet")
st.plotly_chart(comparison_fig)

# ========== STEP 6: CUSTOMER BEHAVIOR INSIGHTS ==========
st.header("3. Customer Behavior Insights")

# Customer Segmentation (RFM Analysis)
st.subheader("1. Customer Segmentation")
# Ensure 'Date' column is in datetime format
sales_data['Date'] = pd.to_datetime(sales_data['Date'])
rfm_data = sales_data.groupby("Customer type").agg({
    "Date": lambda x: (sales_data["Date"].max() - x.max()).days,
    "Invoice ID": "count",
    "Total": "sum"
}).rename(columns={"Date": "Recency", "Invoice ID": "Frequency", "Total": "Monetary"})

rfm_data["Segment"] = pd.qcut(rfm_data["Monetary"], 4, labels=["Low", "Medium", "High", "VIP"])

# Display RFM Data Summary
st.dataframe(rfm_data)

# RFM Visualization
fig_rfm = px.scatter(
    rfm_data, x="Recency", y="Frequency", size="Monetary", color="Segment", 
    title="Customer Segmentation (RFM Analysis)",
    labels={"Recency": "Recency (Days Since Last Purchase)", "Frequency": "Purchase Frequency", "Monetary": "Monetary Value"}
)
st.plotly_chart(fig_rfm)

# Purchase Patterns
st.subheader("2. Purchase Patterns")
purchase_patterns = sales_data.groupby("Product line").agg({
    "Quantity": "sum",
    "Total": "sum"
}).reset_index().sort_values(by="Total", ascending=False)

fig_patterns = px.bar(purchase_patterns, x="Product line", y="Total", color="Product line", 
                      title="Total Sales by Product Line",
                      labels={"Total": "Total Sales", "Product line": "Product Line"})
st.plotly_chart(fig_patterns)

# Churn Analysis
st.subheader("3. Churn Analysis")

# Since 'CustomerID' is not available, we'll use 'Customer type' for analysis
# Calculate the average purchase gap per 'Customer type'
avg_purchase_gap = sales_data.groupby('Customer type')['Date'].apply(lambda x: x.sort_values().diff().mean())

# Convert Timedelta to days
avg_purchase_gap = avg_purchase_gap.dt.days

# Identify 'Customer type' with average purchase gap over 30 days
churn_types = avg_purchase_gap[avg_purchase_gap > 30]

# Display the results
if not churn_types.empty:
    st.write("**At-Risk Customer Types:**")
    for customer_type in churn_types.index:
        gap = avg_purchase_gap[customer_type]
        st.write(f"- {customer_type}: Average purchase gap of {gap:.2f} days.")
else:
    st.write("No customer types with an average purchase gap over 30 days.")

# ========== STEP 7: INSIGHTS ==========
st.header("4. Insights and Recommendations")
st.write("""
- **Top Products**: Focus on promoting strong-performing product lines.
- **Customer Retention**: Offer loyalty programs to VIP customers and re-engage at-risk customers with targeted campaigns.
- **Inventory Optimization**: Stock products based on predicted demand using the Prophet model.
- **Action Plan**: Leverage insights to drive strategic decision-making.
""")
