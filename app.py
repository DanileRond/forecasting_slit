import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np
import plotly.graph_objects as go

# Function to load data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    # Ensure the dataframe has 'ds' and 'y' columns
    if not {'ds', 'y'}.issubset(data.columns):
        st.error("The CSV file must contain 'ds' and 'y' columns.")
        return None
    return data[['ds', 'y']]

# Function to forecast data
def forecast_data(data, periods):
    data.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y'

    # Convert 'ds' to datetime if not already
    data['ds'] = pd.to_datetime(data['ds'])

    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast, model

# Main app
def main():
    st.title('Time Series Forecasting with Prophet Web App')

    uploaded_file = st.file_uploader("Upload your time series CSV", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.success("Data successfully loaded. Please set the forecast parameters below.")

            # Input for forecasting periods
            periods_input = st.number_input(
                'Enter the number of future periods to forecast:',
                min_value=1, max_value=365, value=30, step=1, key='periods_input'
            )

            if st.button('Run Forecast'):
                forecast, model = forecast_data(data, periods_input)

                st.subheader('Forecast Data')
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                st.subheader('Forecast Plot')
                fig1 = plot_plotly(model, forecast)
                st.plotly_chart(fig1)

                st.subheader('Forecast Components')
                fig2 = model.plot_components(forecast)
                st.write(fig2)

                # Plot Actual vs Forecast
                st.subheader('Actual vs Forecasted Values')
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual'))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                st.plotly_chart(fig3)

if __name__ == "__main__":
    main()
