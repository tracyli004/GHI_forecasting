import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# Load your raw data (original scale) and scaled predictions
raw_data = pd.read_csv("data/train_x_old.csv")  # This should be the same data used to fit the scaler
forecast_NHiTS = pd.read_csv("forecast_NHiTS.csv")  # Your predicted values in MinMax scaled form
forecast_TiDE = pd.read_csv("forecast_TiDE.csv") 
forecast_TiDE_RIN = pd.read_csv("forecast_TiDE+RIN.csv") 

# Assuming the target column name is 'target'
y_true_360 = raw_data['Total GHI'][-360:].values.reshape(-1, 1)
y_true_365 = raw_data['Total GHI'][-365:].values.reshape(-1, 1)

forecast_NHiTS_scaled = forecast_NHiTS['NHiTS_Forecast'].values.reshape(-1, 1)
forecast_TiDE_scaled = forecast_TiDE['TiDE_Forecast'].values.reshape(-1, 1)
forecast_TiDE_RIN_scaled = forecast_TiDE_RIN['TiDE+RIN_Forecast'].values.reshape(-1, 1)


# Fit the scaler on original raw data
scaler = MinMaxScaler()
scaler.fit(y_true_360)  # This ensures we get the correct min/max parameters

# Inverse transform the scaled predictions
forecast_TiDE_pred_original = scaler.inverse_transform(forecast_TiDE_scaled)
forecast_TiDE_RIN_pred_original = scaler.inverse_transform(forecast_TiDE_RIN_scaled)

rmse2 = np.sqrt(mean_squared_error(y_true_360, forecast_TiDE_pred_original))
rmse3 = np.sqrt(mean_squared_error(y_true_360, forecast_TiDE_RIN_pred_original))

scaler = MinMaxScaler()
scaler.fit(y_true_365)
forecast_NHiTS_pred_original = scaler.inverse_transform(forecast_NHiTS_scaled)
rmse1 = np.sqrt(mean_squared_error(y_true_365, forecast_NHiTS_pred_original))


print("NHiTS RMSE:", rmse1)
print("TiDE RMSE:", rmse2)
print("TiDE+RIN RMSE:", rmse3)
