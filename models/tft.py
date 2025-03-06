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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint  # ✅ Updated path
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
import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="lightning.pytorch.utilities.parsing")
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # Suppress TensorFlow logs


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
#     # # Baseline MAE: 0.2927

    pl.seed_everything(42)
    class TFTLightningModel(pl.LightningModule):
        def __init__(self, dataset, **hparams):
            super().__init__()
            self.model = TemporalFusionTransformer.from_dataset(dataset, **hparams)
            self.save_hyperparameters(ignore=["loss", "logging_metrics", "dataset"])
            self.dataset_params = dataset.get_parameters()
        
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
        
        def on_save_checkpoint(self, checkpoint):
            checkpoint["dataset_params"] = self.dataset_params  # ✅ Save dataset info in checkpoint

        def on_load_checkpoint(self, checkpoint):
            self.dataset_params = checkpoint["dataset_params"]  # ✅ Reload dataset info
            
        def configure_optimizers(self):
            return torch.optim.Adam(self.parameters(), lr=self.hparams["learning_rate"])  # ✅ Uses saved hyperparams

        def train_dataloader(self):
            return train_dataloader

        def val_dataloader(self):
            return val_dataloader

    hparams = {
        "learning_rate": 0.08906539082147183,
        "lstm_layers": 2,
        "hidden_size": 10,
        "attention_head_size": 1,
        "dropout": 0.2328434370945469,
        "hidden_continuous_size": 8,
        "optimizer": "adam",
    }
    
     # 1. parameters: {'gradient_clip_val': 0.6855404861731059, 
    #              'hidden_size': 10, 
    #              'dropout': 0.2328434370945469, 
    #              'hidden_continuous_size': 8, 
    #              'attention_head_size': 1, 
    #              'learning_rate': 0.08906539082147183}
    
    print(f"Total training batches: {len(train_dataloader)}")
    print(f"Total validation batches: {len(val_dataloader)}")
    print(f"Total test batches: {len(test_dataloader)}")

    # Initialize the TFT Lightning Model
    tft_lightning = TFTLightningModel(training, **hparams)

    # Logging & Callbacks
    lr_logger = LearningRateMonitor()  # Log the learning rate
    logger = TensorBoardLogger("lightning_logs")  # Log results to TensorBoard

    # Early stopping to prevent overfitting
    early_stop_callback = EarlyStopping(
        monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min"
    )

    # Checkpointing to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  
        dirpath="checkpoints/",  
        filename="tft-best-{epoch:02d}-{val_loss:.2f}",  
        save_top_k=1,  
        mode="min"
    )

    # Create PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=50,
        accelerator="gpu",
        enable_model_summary=True,
        gradient_clip_val=0.6855404861731059,
        limit_train_batches=1.0,
        log_every_n_steps=10,
        callbacks=[lr_logger, early_stop_callback, checkpoint_callback],  
        logger=logger,
    )
    

    # trainer.fit(tft_lightning, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader)

    # # Ensure a checkpoint was saved
    # best_model_path = checkpoint_callback.best_model_path
    # if not best_model_path:
    #     raise ValueError("No checkpoint was saved! Ensure ModelCheckpoint is in trainer callbacks.")
    # print(f"Best model saved at: {best_model_path}")

    # # Recreate the dataset before loading the model
    # training = get_training()

    # # Load the best model from the checkpoint
    # best_model_path = "checkpoints/tft-best-epoch=07-val_loss=0.07-v2.ckpt"
    # best_tft = TFTLightningModel.load_from_checkpoint(best_model_path, dataset=training)
    # best_tft.model = TemporalFusionTransformer.from_dataset(training, **best_tft.hparams)
    
    # # Make Predictions
    # predictions = best_tft.model.predict(
    #     val_dataloader, return_y=True, trainer_kwargs=dict(accelerator="cpu")
    # )
    # print("Validation MAE:", MAE()(predictions.output, predictions.y))


    # raw_predictions = best_tft.model.predict(
    #     val_dataloader, mode="raw", return_x=True, trainer_kwargs=dict(accelerator="gpu")
    # )

    # print(f"Total validation batches: {len(val_dataloader)}")
    
    
    # import matplotlib.pyplot as plt
    # import pandas as pd

    # # Plot the TFT model prediction
    # best_tft.model.plot_prediction(
    #     x=raw_predictions.x,  
    #     out=raw_predictions.output,  
    #     idx=0,  
    #     add_loss_to_title=True,
    # )
    # plt.savefig("prediction_plot.png")
    # print("Plot saved as prediction_plot.png")
    
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
        trainer_kwargs=dict(limit_train_batches=30, log_every_n_steps=5),
        reduce_on_plateau_patience=4,
        use_learning_rate_finder=False,  # use Optuna to find ideal learning rate or use in-built learning rate finder
    )

    # save study results - also we can resume tuning at a later point in time
    with open("test_study.pkl", "wb") as fout:
        pickle.dump(study, fout)

    # show best hyperparameters
    print(study.best_trial.params)
    # 1. parameters: {'gradient_clip_val': 0.6855404861731059, 
    #              'hidden_size': 10, 
    #              'dropout': 0.2328434370945469, 
    #              'hidden_continuous_size': 8, 
    #              'attention_head_size': 1, 
    #              'learning_rate': 0.08906539082147183}
    
    
    
    
    
    
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
    
    
    
    







        