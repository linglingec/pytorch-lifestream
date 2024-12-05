import os
import yaml
import logging
import argparse

import torch
import numpy as np
import pandas as pd
import torch.backends.cudnn as cudnn
from tqdm import tqdm
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint

from utils import get_dataloader, InferenceCallback
from nppr_module import NpprPretrainModule, NpprInferenceModule

# Set up logging
logging.basicConfig(
    filename="progress_log.txt",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logging.info("Starting main.py execution.")

# Parse command-line arguments
parser = argparse.ArgumentParser(description="Run the training pipeline.")
parser.add_argument(
    "--config",
    type=str,
    default="config.yaml",
    help="Path to the configuration file (default: config.yaml)"
)
parser.add_argument(
    "--log_file",
    type=str,
    default="progress_log.txt",
    help="Path to the log file (default: progress_log.txt)"
)
args = parser.parse_args()

# Update logging file based on the argument
logging.basicConfig(
    filename=args.log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load configuration
try:
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    logging.info("Configuration loaded successfully from %s.", args.config)
except Exception as e:
    logging.error(f"Failed to load configuration: {e}")
    raise

# Extract configuration groups
data_config = config["data"]
model_config = config["model"]
training_config = config["training"]

# Prepare dataloaders
try:
    train_emb_dataloader = get_dataloader(
        data_config["train_emb_path"],
        data_config["numeric_cols"],
        data_config["categorical_cols"],
        data_config["time_col"],
        training_config["max_past_events"],
        training_config["last_n_transactions"],
        training_config["batch_size"],
        training_config["num_workers"],
        train=True
    )
    train_clf_dataloader = get_dataloader(
        data_config["train_clf_path"],
        data_config["numeric_cols"],
        data_config["categorical_cols"],
        data_config["time_col"],
        training_config["max_past_events"],
        None,
        training_config["batch_size"],
        training_config["num_workers"],
        train=False
    )
    test_clf_dataloader = get_dataloader(
        data_config["test_clf_path"],
        data_config["numeric_cols"],
        data_config["categorical_cols"],
        data_config["time_col"],
        training_config["max_past_events"],
        None,
        training_config["batch_size"],
        training_config["num_workers"],
        train=False
    )
    logging.info("Dataloaders initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize dataloaders: {e}")
    raise

# Enable CUDA optimizations
try:
    cudnn.enabled = True
    cudnn.benchmark = False
    cudnn.deterministic = True
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    logging.info("CUDA optimizations enabled.")
except Exception as e:
    logging.error(f"Failed to enable CUDA optimizations: {e}")
    raise

# Initialize pretrain module
try:
    model = NpprPretrainModule(
        embedding_dims=model_config["embedding_dims"],
        embedding_size=model_config["embedding_size"],
        hidden_size_enc=model_config["hidden_size_enc"],
        hidden_size_dec=model_config["hidden_size_dec"],
        num_numerical_features=model_config["num_numerical_features"],
        num_categories=model_config["num_categories"],
        learning_rate=model_config["learning_rate"],
        total_steps=model_config["total_steps"],
        pct_start=model_config["pct_start"],
        lambda_param=model_config["lambda_param"],
        alpha=model_config["alpha"]
    )
    logging.info("Model initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize model: {e}")
    raise

# Define callbacks
try:
    checkpoint_callback = ModelCheckpoint(
        dirpath="checkpoints_past-events=50",
        filename="nppr-{epoch:02d}",
        save_top_k=-1,
        save_weights_only=True,
        every_n_epochs=1,
        save_last=True
    )
    logging.info(f"ModelCheckpoint initialized.")
    logging.info(f"ModelCheckpoint dirpath: {checkpoint_callback.dirpath}")
    logging.info(f"Save top_k: {checkpoint_callback.save_top_k}")

    inference_callback = InferenceCallback(
        checkpoint_callback=checkpoint_callback,
        train_clf_dataloader=train_clf_dataloader,
        test_clf_dataloader=test_clf_dataloader,
        train_clf_path=data_config["train_clf_path"],
        test_clf_path=data_config["test_clf_path"],
        train_target_path=data_config["train_target_path"],
        results_path=model_config["results_path"],
        embedding_dims=model_config["embedding_dims"],
        embedding_size=model_config["embedding_size"],
        hidden_size_enc=model_config["hidden_size_enc"],
        hidden_size_dec=model_config["hidden_size_dec"],
        num_numerical_features=model_config["num_numerical_features"],
        num_categories=model_config["num_categories"],
        inference_device="cuda:1"
    )
    logging.info("InferenceCallback initialized successfully.")
    logging.info("Callbacks defined successfully.")
except Exception as e:
    logging.error(f"Failed to define callbacks: {e}")
    raise

# Set up trainer
try:
    trainer = Trainer(
        max_epochs=training_config["num_epochs"],
        devices=[0],
        accelerator="gpu",
        callbacks=[checkpoint_callback, inference_callback],
        enable_checkpointing=True
    )
    logging.info("Trainer initialized successfully.")
except Exception as e:
    logging.error(f"Failed to initialize trainer: {e}")
    raise

# Train the model
try:
    logging.info("Starting training...")
    trainer.fit(model, train_emb_dataloader)
    logging.info("Training completed successfully.")
except Exception as e:
    logging.error(f"Training failed: {e}")
    raise
