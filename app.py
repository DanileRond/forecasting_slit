import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly
import numpy as np

# Function to load data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)[['ds','y']]
    return data

def forecast_data(data, periods):
    data.columns = ['ds', 'y']  # Prophet expects 'ds' and 'y'
    model = Prophet()
    model.fit(data)

    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    # Calculate forecast error on the test set if exists
    if len(forecast) > len(data):
        forecasted = forecast.iloc[-periods:]
        actual = data.iloc[-periods:]['y']
        forecast_error = ((forecasted['yhat'] - actual).abs() / actual).mean()

        st.write(f"Mean Absolute Percentage Error (MAPE) on the test set: {forecast_error:.2%}")
    return forecast, model
# Main app
def main():
    st.title('Time Series Forecasting with Prophet WEBAPP')
    
    uploaded_file = st.file_uploader("Upload your time series CSV", type="csv")

    if uploaded_file is not None:
        data = load_data(uploaded_file)
        st.success("Data successfully loaded. Please set the forecast parameters below.")
        
        # Only allow the user to set periods and run the forecast after uploading the data
        periods_input = st.number_input('Enter the number of future periods to forecast:', min_value=1, max_value=365, key='periods_input')
        
        if st.button('Run Forecast'):
            forecast, model = forecast_data(data, periods_input)
            
            st.subheader('Forecast data')
            st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

            st.subheader("Forecast plot")
            fig1 = plot_plotly(model, forecast)
            st.plotly_chart(fig1)

            if len(forecast) > len(data):
                st.subheader("Forecast error plot")
                fig2 = plot_plotly(model, forecast)  # Reuse plot with forecast details
                fig2.data[0].name = 'Forecast'
                fig2.add_scatter(x=data['ds'], y=data['y'], mode='lines+markers', name='Actual', line=dict(color='red'))
                st.plotly_chart(fig2)

if __name__ == "__main__":
    main()
