import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler

# Load dataset
data_df = pd.read_csv("data/train_x_old.csv")

# Ensure a time_idx column: Convert datetime feature into an increasing numerical index
# Ensure data is sorted correctly
data_df = data_df.sort_values(["Year", "Month", "Day"]).reset_index(drop=True)
data_df["time_idx"] = range(len(data_df))
# data_df["time_idx"] = (data_df["Year"] - data_df["Year"].min()) * 365 + data_df["Day"]



# Convert categorical variables to category dtype
data_df["Year"] = data_df["Year"].astype("category")
data_df["Month"] = data_df["Month"].astype("category")
data_df["Day"] = data_df["Day"].astype("category")

# Apply log transformations to stabilize variance for skewed variables
log_cols = ["Average Ozone", "Average AOD", "Total GHI"]
for col in log_cols:
    data_df[col] = np.log(data_df[col] + 1e-8)

# Create grouped statistical features (mean by time index)
grouped_features = {
    "Average Temperature": "avg_temp_by_month",
    "Average Relative Humidity": "avg_humidity_by_month"
}
for col, new_col in grouped_features.items():
    data_df[new_col] = data_df.groupby(["time_idx", "Month"])[col].transform("mean")

# One-hot encode Daylight Weather and convert into a single categorical column
weather_cols = ['Daylight Weather_Clear', 'Daylight Weather_Cloudy',
                'Daylight Weather_Fog', 'Daylight Weather_Partly Cloudy']

data_df["weather"] = data_df[weather_cols].idxmax(axis=1)  # Get column with max value
data_df["weather"] = data_df["weather"].astype(str).str.replace("Daylight Weather_", "")
data_df["weather"] = data_df["weather"].astype("category")


# Normalize continuous features
scaler = MinMaxScaler()
continuous_cols = ["Average Temperature", "Average Dew Point", "Average Relative Humidity", 
                   "Average Pressure", "Average Precipitable Water", "Max Temperature", 
                   "Min Temperature", "RMM2", "MJO Amplitude"]
data_df[continuous_cols] = scaler.fit_transform(data_df[continuous_cols])

# Drop redundant columns
data_df.drop(columns=weather_cols, inplace=True)

# Save processed data
data_df.to_csv("data/train_x.csv", index=False)

# Display processed data
print(data_df.head())
print(data_df.columns)
