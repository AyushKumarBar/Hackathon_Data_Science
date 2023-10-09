import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

# Prepare the data
data = {
    'Year': [1993, 1997, 2002, 2007, 2012, 2017, 2018, 2019, 2020, 2021, 2022],
    'Population': [25569, 25842, 26373, 27669, 29391, 29964, 29607, 29301, 29039, 28814, 28622]
}
df = pd.DataFrame(data)

# Fit the ARIMA model
model = ARIMA(df['Population'], order=(1, 0, 0))
model_fit = model.fit()

# Predict for the years 2018 to 2022
predictions = model_fit.predict(start=len(df), end=len(df) + 2)

# Print the predicted values
print(predictions)




