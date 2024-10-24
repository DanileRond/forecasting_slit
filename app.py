import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_cross_validation_metric
from prophet.diagnostics import cross_validation, performance_metrics
import plotly.graph_objects as go
import itertools

# Function to load data
def load_data(uploaded_file):
    data = pd.read_csv(uploaded_file)
    if not {'ds', 'y'}.issubset(data.columns):
        st.error("The CSV file must contain 'ds' and 'y' columns.")
        return None
    return data[['ds', 'y']]

# Function to forecast data with Prophet and perform hyperparameter tuning
def forecast_data_prophet(data, periods):
    data['ds'] = pd.to_datetime(data['ds'])
    
    # Tuning parameters
    param_grid = {
        'changepoint_prior_scale': [0.001, 0.01, 0.1, 0.5],
        'seasonality_prior_scale': [0.01, 0.1, 1.0, 10.0],
    }
    
    all_params = [dict(zip(param_grid.keys(), v)) for v in itertools.product(*param_grid.values())]
    rmses = []  # Store the RMSEs for each params

    # Use cross validation to evaluate all parameters
    for params in all_params:
        m = Prophet(**params).fit(data)  # Fit model with given params
        df_cv = cross_validation(m, horizon='30 days', parallel="processes")
        df_p = performance_metrics(df_cv, rolling_window=1)
        rmses.append(df_p['rmse'].values[0])

    # Find the best parameters
    best_params = all_params[np.argmin(rmses)]

    # Fit model with the best parameters
    model = Prophet(**best_params)
    model.fit(data)
    future = model.make_future_dataframe(periods=periods)
    forecast = model.predict(future)

    return forecast, model, best_params

# Main app
def main():
    st.title('Forecasting App')
    uploaded_file = st.file_uploader("Upload your time series CSV", type="csv")
    if uploaded_file is not None:
        data = load_data(uploaded_file)
        if data is not None:
            st.success("Data successfully loaded. Please set the forecast parameters below.")
            periods_input = st.number_input('Enter the number of future periods to forecast:', min_value=1, max_value=365, value=30, step=1)
            
            if st.button('Run Prophet Forecast'):
                forecast, model, best_params = forecast_data_prophet(data, periods_input)
                st.subheader('Best Model Parameters')
                st.write(best_params)
                
                st.subheader('Forecast Data')
                st.write(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail())

                st.subheader('Forecast Plot')
                fig1 = plot_plotly(model, forecast)
                st.plotly_chart(fig1)

                # Plot Actual vs Forecast
                st.subheader('Actual vs Forecasted Values')
                fig3 = go.Figure()
                fig3.add_trace(go.Scatter(x=data['ds'], y=data['y'], mode='lines', name='Actual'))
                fig3.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], mode='lines', name='Forecast'))
                st.plotly_chart(fig3)

if __name__ == "__main__":
    main()
