import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# Load the data into a pandas DataFrame
data = pd.read_excel('1-00S968-L84-140-140.xlsx')
df = pd.DataFrame(data)
df['MONTH_YEAR'] = pd.to_datetime(df['MONTH_YEAR'])
df.set_index('MONTH_YEAR', inplace=True)

# Fit the ARIMA model
model = ARIMA(df, order=(1, 1, 1))
model_fit = model.fit()

# Forecast one year ahead
forecast = model_fit.get_forecast(steps=12)

# Extract the forecasted values and confidence intervals
forecast_values = forecast.predicted_mean
confidence_intervals = forecast.conf_int()

# Plot the forecasted values and confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['QUANTITY'], label='Actual')
plt.plot(forecast_values.index, forecast_values, label='Forecast')
plt.fill_between(confidence_intervals.index, confidence_intervals.iloc[:, 0], confidence_intervals.iloc[:, 1],
                 color='gray', alpha=0.3, label='Confidence Intervals')
plt.title('ARIMA Forecast')
plt.xlabel('Month')
plt.ylabel('Quantity')
plt.legend()
plt.show()
