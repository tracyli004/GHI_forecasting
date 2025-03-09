import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import copy
from pathlib import Path
import warnings
import pickle
import os

from pytorch_forecasting import TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer

from darts.models import NHiTSModel, TiDEModel
from darts.metrics import mape
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from darts.datasets import AusBeerDataset
from darts.metrics import mae, mse
from darts.models import NHiTSModel, TiDEModel

import logging
logging.disable(logging.CRITICAL)
warnings.filterwarnings("ignore")

import torch
torch.set_default_dtype(torch.float32)






optimizer_kwargs = {
    "lr": 1e-3,
}
# PyTorch Lightning Trainer arguments
pl_trainer_kwargs = {
    "gradient_clip_val": 1,
    "max_epochs": 100,
    "accelerator": "auto",
    "callbacks": [],
}
# learning rate scheduler
lr_scheduler_cls = torch.optim.lr_scheduler.ExponentialLR
lr_scheduler_kwargs = {
    "gamma": 0.999,
}
# early stopping (needs to be reset for each model later on)
# this setting stops training once the the validation loss has not decreased by more than 1e-3 for 10 epochs
early_stopping_args = {
    "monitor": "val_loss",
    "patience": 10,
    "min_delta": 1e-3,
    "mode": "min",
}
common_model_args = {
    "input_chunk_length": 365,  # lookback window
    "output_chunk_length": 365,  # forecast/lookahead window
    "optimizer_kwargs": optimizer_kwargs,
    "pl_trainer_kwargs": pl_trainer_kwargs,
    "lr_scheduler_cls": lr_scheduler_cls,
    "lr_scheduler_kwargs": lr_scheduler_kwargs,
    "likelihood": None,  # use a likelihood for probabilistic forecasts
    "save_checkpoints": True,  # checkpoint to retrieve the best performing model state,
    "force_reset": True,
    "batch_size": 256,
    "random_state": 42

}

data_df = pd.read_csv("data/train_x.csv")
data_df["date"] = pd.to_datetime(data_df[["Year", "Month", "Day"]])
data_df.set_index("date", inplace=True)

# convert to series, fill NA, convert back to series
series = TimeSeries.from_dataframe(data_df, value_cols="Total GHI", freq="D")
df_series = series.pd_dataframe()
df_series_filled = df_series.fillna(method="ffill").fillna(method="bfill")
series = TimeSeries.from_dataframe(df_series_filled, value_cols="Total GHI", freq="D")
print("Total Missing Values in TimeSeries:", series.pd_dataframe().isna().sum().sum())

# Split train, val, test
train, temp = series.split_after(0.6)
val, test = temp.split_after(0.5)

train = train.astype(np.float32)
val = val.astype(np.float32)


# create the models
model_nhits = NHiTSModel(**common_model_args, model_name="hi")
model_tide = TiDEModel(
    **common_model_args, use_reversible_instance_norm=False, model_name="tide0"
)
model_tide_rin = TiDEModel(
    **common_model_args, use_reversible_instance_norm=True, model_name="tide1"
)
models = {
    "NHiTS": model_nhits,
    "TiDE": model_tide,
    "TiDE+RIN": model_tide_rin,
}

# Define hyperparameter grid
param_grid = {
    "input_chunk_length": [365, 545, 730],  # History window (try 1 yr, 1.5, 2)
    "output_chunk_length": [90, 120, 365],
    "n_epochs": [50, 100],  # Number of training epochs
    "batch_size": [64, 128],  # Batch size for training
    "dropout": [0.1, 0.3],  # Dropout rate for regularization
}

# Load previously saved models if they exist
save_path = "best_models.pkl"
if os.path.exists(save_path):
    with open(save_path, "rb") as f:
        best_models = pickle.load(f)
else:
    best_models = {}

    # Loop through models and run grid search
for name, model in models.items():
    if name in best_models:  # Skip already processed models
        print(f"Skipping {name}, already completed.")
        continue
    
    print(f"Running Grid Search for {name}...")

    # Perform grid search
    best_model, best_params, best_score = model.gridsearch(
        parameters=param_grid,  # Hyperparameter grid
        series=train,
        val_series=val,
        metric=mape,
        verbose=True
    )

    # Store results
    best_models[name] = {
        "model": best_model,
        "params": best_params,
        "score": best_score
    }
    with open(save_path, "wb") as f:
        pickle.dump(best_models, f)

    print(f"✅ Best Model for {name}: {best_model}")
    print(f"🔹 Best Params: {best_params}")
    print(f"📉 Best Score: {best_score}")


# Train with grid searched params
for name, model in best_models.items():
    pl_trainer_kwargs["callbacks"] = [EarlyStopping(**early_stopping_args)]
    best_model = model["model"]
    print(f"Training best {name} model...")
    best_model.fit(train, val_series=val, verbose=True)

    print(f"{name} Trained for {best_model.trainer.current_epoch} epochs")


# Test and plot
result_accumulator = {}
# Create separate plots for each model
for name, model in best_models.items():
    fig, ax = plt.subplots(figsize=(15, 5))  # Create a new figure for each model

    n_steps = 365  # Forecast one full year
    pred_steps_per_chunk = model["model"].output_chunk_length
    # pred_steps_per_chunk = model.output_chunk_length  # Model predicts in chunks (e.g., 90 days)

    input_series = test[: -n_steps]  # Start with the last known real data
    rolling_predictions = []
    
    input_series = input_series.astype("float32")
    for _ in range(n_steps // pred_steps_per_chunk):
        pred = model["model"].predict(n=pred_steps_per_chunk, series=input_series)
        rolling_predictions.append(pred)
        input_series = input_series.append(pred)  # Feed predictions back into the model

    # Merge all predictions into one time series
    full_year_forecast = rolling_predictions[0]
    for pred in rolling_predictions[1:]:
        full_year_forecast = full_year_forecast.concatenate(pred)
        
        
    forecast_df = full_year_forecast.pd_dataframe()
    forecast_df.columns = [f"{name}_Forecast"]
    csv_save_path = f"forecast_{name}.csv"
    forecast_df.to_csv(csv_save_path)
    print(f"Saved predictions for {name} as {csv_save_path}")

    # Plot forecast and actual data
    test[-365:].plot(label="Actual", ax=ax, color='#1f77b4', linewidth=1)
    full_year_forecast.plot(label=f"{name} Forecast", ax=ax, color='#ff7f0e', linewidth=1)

    # Customize the plot
    ax.legend()
    ax.set_title(f"{name} Model Forecast vs. Actual")
    ax.set_xlabel("Time")
    ax.set_ylabel("Total GHI")

    # Store MAE & MSE results
    result_accumulator[name] = {
        "mae": mae(test, full_year_forecast),
        "mse": mse(test, full_year_forecast),
    }
    print(f"{name} MSE:", result_accumulator[name]["mse"])

    save_path = f"forecast_plot_{name}.png"
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"Saved plot for {name} as {save_path}")

    plt.close(fig)  # Close the figure to free memory