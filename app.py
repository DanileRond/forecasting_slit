import streamlit as st
import pandas as pd
import numpy
from prophet import Prophet
from prophet.plot import plot_plotly
import plotly.graph_objects as go
from neuralforecast.auto import AutoTiDE
from neuralforecast.losses.pytorch import SMAPE
from neuralforecast import NeuralForecast

# Function to load data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    # Ensure the dataframe has 'ds' and 'y' columns
    if not {'ds', 'y'}.issubset(data.columns):
        st.error("The CSV file must contain 'ds' and 'y' columns.")
        return None
    return data[['ds', 'y']]

# Function to forecast data with Prophet
def forecast_data_prophet(data, periods):
    data.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y'
    data['ds'] = pd.to_datetime(data['ds'])
    model = Prophet()
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)
    return forecast, model

# Function to forecast data with AutoTiDE
def forecast_data_autotide(data, periods):
    # Ensure the DataFrame column names are as expected by AutoTiDE
    data = data.rename(columns={'ds': 'Date', 'y': 'Value'})
    data['Date'] = pd.to_datetime(data['Date'])  # Ensure 'Date' is in datetime format

    smape_loss = SMAPE()
    model = AutoTiDE(h=periods, loss=smape_loss)
    fcst = NeuralForecast(models=[model], freq='D')  # Adjust frequency if needed

    # Preparing data for NeuralForecast which expects 'ds' and 'y' columns
    data = data.rename(columns={'Date': 'ds', 'Value': 'y'})
    data['unique_id'] = 'series1'  # Assign a constant value to all rows
    forecast = fcst.cross_validation(data, n_windows=1, step_size=periods, refit=True)
    return forecast, model

# Main app
def main():
    st.title('Forecasting App')
    uploaded_file = st.file_uploader("Upload your time series CSV", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.success("Data successfully loaded. Please set the forecast parameters below.")
            periods_input = st.number_input('Enter the number of future periods to forecast:', min_value=1, max_value=365, value=30, step=1, key='periods_input')
            
            if st.button('Run Prophet Forecast'):
                forecast_p, model_p = forecast_data_prophet(data, periods_input)
                st.subheader('Prophet Forecast Data')
                st.write(forecast_p[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())
                fig1 = plot_plotly(model_p, forecast_p)
                st.plotly_chart(fig1)

            if st.button('Run AutoTiDE Forecast'):
                forecast_t, model_t = forecast_data_autotide(data, periods_input)
                st.subheader('AutoTiDE Forecast Data')
                st.write(forecast_t)
                # Add additional plotting code if necessary

if __name__ == "__main__":
    main()
