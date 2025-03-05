import copy
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
import pickle
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

# Lightning imports
import lightning.pytorch as pl
from lightning.pytorch.tuner import Tuner  # ✅ Correct
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor  # ✅ Updated path
from lightning.pytorch.loggers import TensorBoardLogger  # ✅ Updated path

# PyTorch Forecasting imports
from pytorch_forecasting import Baseline, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data import GroupNormalizer
from pytorch_forecasting.metrics import MAE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.data.encoders import NaNLabelEncoder  # ✅ Correct

import tensorflow as tf
import tensorboard as tb

tf.io.gfile = tf.compat.v1.io.gfile

# Example dataset (for testing models)
# You likely don't need this unless you're running example scripts.
# from pytorch_forecasting.data.examples import get_stallion_data  # Optional


from data.dataloading import get_training, get_dataloaders


if __name__ == "__main__":

    # Load dataloaders
    train_dataloader = torch.load("data/train_dataloader.pth", weights_only=False)
    val_dataloader = torch.load("data/val_dataloader.pth", weights_only=False)
    test_dataloader = torch.load("data/test_dataloader.pth", weights_only=False)
    # train_dataloader, val_dataloader, test_dataloader = get_dataloaders()
    training = get_training()

#     # # Run Baseline model
#     # baseline_predictions = Baseline().predict(val_dataloader, return_y=True)
#     # baseline_mae = MAE()(baseline_predictions.output, baseline_predictions.y)
#     # print("Baseline MAE:", baseline_mae)
#     # # Baseline MAE: tensor(0.2927, device='mps:0')

    pl.seed_everything(42)
    class TFTLightningModel(pl.LightningModule):
        def __init__(self, dataset, **hparams):
            super().__init__()
            self.model = TemporalFusionTransformer.from_dataset(dataset, **hparams)
            self.save_hyperparameters(ignore=["loss", "logging_metrics"])
        
        def training_step(self, batch, batch_idx):
            x, y = batch  # x: input features, y: ground truth target
            x = {k: v.to(self.device) for k, v in x.items()}  # Move input data to MPS
            y = tuple(v.to(self.device) if v is not None else None for v in y) 

            output = self.model(x)  # TFT returns a structured output
            y_hat = output.prediction  # Named tuple attribute instead of dictionary key
            loss = self.model.loss(y_hat, y)
            
            batch_size = x["encoder_lengths"].shape[0]
            self.log("train_loss", loss, batch_size=batch_size)
            return loss

        def validation_step(self, batch, batch_idx):
            x, y = batch
            x = {k: v.to(self.device) for k, v in x.items()}  # Move input data to MPS
            y = tuple(v.to(self.device) if v is not None else None for v in y) 
                       
            output = self.model(x)
            y_hat = output.prediction
            loss = self.model.loss(y_hat, y)
            batch_size = x["encoder_lengths"].shape[0]
            self.log("val_loss", loss, prog_bar=True, batch_size=batch_size)
            return loss

        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])  # ✅ Uses saved hyperparams

        def train_dataloader(self):
            return train_dataloader

        def val_dataloader(self):
            return val_dataloader



    hparams = {
        "learning_rate": 0.004365158322401661,
        "hidden_size": 8,
        "attention_head_size": 1,
        "dropout": 0.1,
        "hidden_continuous_size": 8,
        "optimizer": "adam",
    }

    tft_lightning = TFTLightningModel(training, **hparams)
    # configure network and trainer
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )
    lr_logger = LearningRateMonitor()  # log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # logging results to a tensorboard

    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="mps",
        enable_model_summary=True,
        gradient_clip_val=0.1,
        limit_train_batches=50,  # comment in for training, running validation every 30 batches
        # fast_dev_run=True,  # comment in to check that networkor dataset has no serious bugs
        callbacks=[lr_logger, early_stop_callback],
        logger=logger,
    )
    trainer.fit(
        tft_lightning,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


    # create study
    study = optimize_hyperparameters(
        train_dataloader,
        val_dataloader,
        model_path="optuna_test",
        n_trials=200,
        max_epochs=50,
        gradient_clip_val_range=(0.01, 1.0),
        hidden_size_range=(8, 128),
        hidden_continuous_size_range=(8, 128),
        attention_head_size_range=(1, 4),
        learning_rate_range=(0.001, 0.1),
        dropout_range=(0.1, 0.3),
        trainer_kwargs=dict(limit_train_batches=30),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)
    
    # tuner = Tuner(trainer)
    # try:
    #     res = tuner.lr_find(
    #         tft_lightning,  # LightningModule
    #         train_dataloaders=train_dataloader,
    #         val_dataloaders=val_dataloader,
    #         min_lr=1e-6,
    #         max_lr=10.0,
    #         num_training=100,
    #         early_stop_threshold=None
    #     ) # best lr is 0.004365158322401661
    #     print(f"Suggested Learning Rate: {res.suggestion()}")

    #     fig = res.plot(suggest=True)  # `suggest=True` will mark the suggested LR
    #     fig.show()
    #     fig.savefig("lr_finder_plot.png")
    # except Exception as e:
    #     print(f"LR Finder stopped early: {e}")
    
    
    
    







        