import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

# Load preprocessed data
data_df = pd.read_csv("data/train_x.csv")

# Convert Year, Month, and Day to categorical dtype
# data_df["Year"] = data_df["Year"].astype(str).astype("category")
data_df["Month"] = data_df["Month"].astype(str).astype("category")
data_df["Day"] = data_df["Day"].astype(str).astype("category")
data_df["Location"] = "22.03386, -159.78313"
# Ensure target column is a float
data_df["Total GHI"] = data_df["Total GHI"].astype(float)

data_df["Year"] = (data_df["Year"].astype(int) - data_df["Year"].astype(int).min()) / (
    data_df["Year"].astype(int).max() - data_df["Year"].astype(int).min()
)


# Define max encoder & prediction length
max_prediction_length = 365  # Rolling forecast
max_encoder_length = 365  # Use past 2 yrs to make predictions

train_cutoff = data_df["time_idx"].max() - (4*365)  # Training stops 2 years before last
val_cutoff = data_df["time_idx"].max() - (365*2)  # Validation stops 1 year before last

train_df = data_df[data_df["time_idx"] <= train_cutoff]
val_df = data_df[(data_df["time_idx"] > train_cutoff) & (data_df["time_idx"] <= val_cutoff)]
test_df = data_df[data_df["time_idx"] > val_cutoff]  # Last year for testing

# print(f"Train size: {len(train_df)}, Val size: {len(val_df)}, Test size: {len(test_df)}")
# print("Min time idx: ", data_df["time_idx"].min(), "Max time idx: ", data_df["time_idx"].max())
# print("Train:", train_df["time_idx"].min(), train_df["time_idx"].max())
# print("Val:", val_df["time_idx"].min(), val_df["time_idx"].max())
# print("Test:", test_df["time_idx"].min(), test_df["time_idx"].max())


# Define training dataset
training = TimeSeriesDataSet(
    data_df[lambda x: x.time_idx <= train_cutoff],
    time_idx="time_idx",
    target="Total GHI",  # Predict daily temperatures
    group_ids=["Location"],
    min_encoder_length=max_encoder_length // 2,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,

    static_categoricals=[],  # No static categorical features
    static_reals=[],

    # 🔥 Year, Month, and Day are known ahead of time
    time_varying_known_categoricals=["Month", "Day"],  # Known in advance
    time_varying_known_reals=["Year", "time_idx"],  # Known in advance


    time_varying_unknown_categoricals=["weather"],  # Model must learn daily weather changes
    time_varying_unknown_reals=[
        "Average Temperature", "Average Ozone", "Average AOD",
        "Average Dew Point", "Average Relative Humidity", "Average Pressure",
        "Average Precipitable Water", "Max Temperature", "Min Temperature",
        "RMM2", "MJO Amplitude", "avg_temp_by_month", "avg_humidity_by_month"
    ],
    
    target_normalizer=GroupNormalizer(groups=["Location"], transformation="softplus"),  # ✅ Normalize separately per year
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
    allow_missing_timesteps=True,
    # categorical_encoders={
    #     # "Year": TorchCategoricalEncoder(add_nan=True)
    #     "Year": NaNLabelEncoder(add_nan=True)  # Allow unseen years like 2022/2023 because we only train 2000-2021/12/31
    # }
)


# Create validation dataset
validation = TimeSeriesDataSet.from_dataset(
    training, val_df, predict=True, stop_randomization=True
)
test = TimeSeriesDataSet.from_dataset(
    training, test_df, predict=True, stop_randomization=True
)

# Convert to dataloaders
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=10, persistent_workers=True)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=10, persistent_workers=True)
test_dataloader = test.to_dataloader(train=False, batch_size=batch_size * 10, num_workers=10, persistent_workers=True)

# print("DataLoader setup complete!")

# Save dataloaders
torch.save(train_dataloader, "data/train_dataloader.pth")
torch.save(val_dataloader, "data/val_dataloader.pth")
torch.save(test_dataloader, "data/test_dataloader.pth")

# print("Dataloaders saved successfully!")


def get_training():
    return training

def get_dataloaders():
    return train_dataloader, val_dataloader, test_dataloader
