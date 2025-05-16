import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Dummy data for time series
data = pd.DataFrame({
    'Date': pd.date_range('2023-01-01', periods=100),
    'Value': [15, 12, 13, 10, 9, 12, 14, 18, 16, 19, 22, 25, 28, 30, 32, 35, 38, 40, 42, 45,
              48, 51, 53, 56, 59, 62, 65, 68, 71, 74, 77, 80, 83, 86, 89, 92, 95, 98, 101,
              104, 107, 110, 113, 116, 119, 122, 125, 128, 131, 134, 137, 140, 143, 146, 149,
              152, 155, 158, 161, 164, 167, 170, 173, 176, 179, 182, 185, 188, 191, 194, 197,
              200, 203, 206, 209, 212, 215, 218, 221, 224, 227, 230, 233, 236, 239, 242, 245,
              248, 251, 254, 257, 260, 263, 266, 269, 272, 275, 278]
})

# App title and description
st.title('ARIMA Model')
st.write('This app fits an ARIMA model to a time series and makes predictions.')

# Display the time series data
st.subheader('Time Series Data')
st.dataframe(data)

# Fit ARIMA model
model = ARIMA(data['Value'], order=(2, 1, 1))  # Example order (p, d, q)
model_fit = model.fit()

# Make predictions
future_periods = st.slider('Future Periods', 1, 10, 5)
forecast, stderr, conf_int = model_fit.forecast(steps=future_periods)

# Display forecasted values
st.subheader('Forecasted Values')
forecast_data = pd.DataFrame({
    'Date': pd.date_range(start=data['Date'].iloc[-1] + pd.DateOffset(days=1), periods=future_periods),
    'Forecast': forecast
})
st.dataframe(forecast_data)

# Plot the time series and forecast
st.subheader('Time Series with Forecast')
plt.figure(figsize=(10, 6))
plt.plot(data['Date'], data['Value'], label='Time Series')
plt.plot(forecast_data['Date'], forecast_data['Forecast'], label='Forecast')
plt.xlabel('Date')
plt.ylabel('Value')
plt.title('ARIMA Model')
plt.legend()
st.pyplot(plt)
