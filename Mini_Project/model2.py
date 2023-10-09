import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import STL

# Load the data into a pandas DataFrame (same as above)
data = pd.read_excel('1-00S968-L84-140-140.xlsx')
df = pd.DataFrame(data)
df['MONTH_YEAR'] = pd.to_datetime(df['MONTH_YEAR'])
df.set_index('MONTH_YEAR', inplace=True)
# Decompose the time series using STL
stl = STL(df)
result = stl.fit()

# Generate one-year forecasts
n_forecast = 12
seasonal_period = 12  # Assuming a seasonal period of 12 months
last_seasonal = result.seasonal[-seasonal_period:].values
trend_forecast = result.trend[-n_forecast:].values
seasonal_forecast = np.tile(
    last_seasonal, n_forecast // seasonal_period + 1)[:n_forecast]
residual_forecast = result.resid[-n_forecast:].values

# Combine the seasonal forecasts with the trend and residual components
forecast_values = trend_forecast + seasonal_forecast + residual_forecast

# Plot the forecasted values
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['QUANTITY'], label='Actual')
plt.plot(pd.date_range(
    start=df.index[-1], periods=n_forecast, freq='M'), forecast_values, label='Forecast')
plt.title('STL Forecast')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.legend()
plt.show()
