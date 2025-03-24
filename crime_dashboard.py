import streamlit as st
import pandas as pd
import plotly.express as px

# Load the transformed dataset
DATA_FILE = "crime_data_transformed.csv"
crime_df = pd.read_csv(DATA_FILE)

# Convert date components to proper data types
crime_df["Year"] = crime_df["Year"].astype(int)
crime_df["Month"] = crime_df["Month"].astype(int)
crime_df["Day"] = crime_df["Day"].astype(int)

# Create a DateTime column
crime_df["DateTime"] = pd.to_datetime(crime_df[["Year", "Month", "Day"]])

# Streamlit app
st.title("Crime Data Dashboard")

# View selection
view_option = st.radio("Select View Mode:", ["Year", "Month", "Day", "All"], index=3)

# Filter data based on view selection
if view_option == "Year":
    crime_agg = crime_df.groupby("Year").size().reset_index(name="Crime Count")
    fig = px.bar(crime_agg, x="Year", y="Crime Count", title="Crime Count by Year")
elif view_option == "Month":
    crime_agg = crime_df.groupby("Month").size().reset_index(name="Crime Count")
    fig = px.bar(crime_agg, x="Month", y="Crime Count", title="Crime Count by Month")
elif view_option == "Day":
    crime_agg = crime_df.groupby("Day").size().reset_index(name="Crime Count")
    fig = px.bar(crime_agg, x="Day", y="Crime Count", title="Crime Count by Day")
else:
    crime_agg = crime_df.groupby("DateTime").size().reset_index(name="Crime Count")
    fig = px.line(
        crime_agg, x="DateTime", y="Crime Count", title="Crime Trends Over Time"
    )

# Display plot
st.plotly_chart(fig)
