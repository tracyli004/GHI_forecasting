import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler


# MJO indices
mjo_url = "https://psl.noaa.gov/mjo/mjoindex/vpm.1x.txt"

mjo_df = pd.read_csv(mjo_url, delim_whitespace=True, header=None,
                     names=['Year', 'Month', 'Day', 'Hour', 'RMM1', 'RMM2', 'MJO Amplitude']
                    ).drop(columns=['Hour'])
mjo_df = mjo_df[(mjo_df["Year"] >= 2000) & (mjo_df["Year"] <= 2023)]

# Climate data from Hawaii training location
train_loc = "22.03386, -159.78313"
ghi_df = pd.read_csv(f"train_x_locs/{train_loc}.csv").drop(columns=["Date", "Total DHI", "Total DNI",
    "Total Clearsky GHI", "Total Clearsky DHI", "Total Clearsky DNI",])
ghi_df = ghi_df[ghi_df["Year"] != 2002]

# Year, month, day columns
mjo_df["Date"] = pd.to_datetime(mjo_df[["Year", "Month", "Day"]])
ghi_df["Date"] = pd.to_datetime(ghi_df[["Year", "Month", "Day"]])

# Merge MJO and climate data into full dataset
data_df = pd.merge(ghi_df, mjo_df[["Date", "RMM2", "MJO Amplitude"]], on="Date", how="left")
data_df.drop(columns=["Date"], inplace=True)


# Preprocess data --- One-hot encode Daylight Weather and normalize other continuous numerical values

encoder = OneHotEncoder(sparse_output=False)
weather_encoded = encoder.fit_transform(data_df[['Daylight Weather']])
weather_cols = encoder.get_feature_names_out(['Daylight Weather'])
weather_df = pd.DataFrame(weather_encoded, columns=weather_cols)

# Drop original 'Daylight Weather' and add one-hot encoded columns
data_df = pd.concat([data_df.drop(columns=['Daylight Weather']), weather_df], axis=1)

# Normalize continuous features
scaler = MinMaxScaler()
continuous_cols = [
    "Average Temperature", "Average Ozone", "Average AOD", "Average Dew Point",
    "Average Relative Humidity", "Average Pressure", "Average Precipitable Water",
    "Max Temperature", "Min Temperature",
    "RMM2", "MJO Amplitude"
]
data_df[continuous_cols] = scaler.fit_transform(data_df[continuous_cols])

# Set date as type=category
data_df["Year"] = data_df["Year"].astype("category")
data_df["Month"] = data_df["Month"].astype("category")
data_df["Day"] = data_df["Day"].astype("category")

#print(data_df.head(10))

data_df.to_csv('data/train_x.csv', index=False)

