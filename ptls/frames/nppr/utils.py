import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning.callbacks import Callback
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
import logging
import os
from nppr_module import NpprInferenceModule
from nppr_dataset import NPPRDataset
from nppr_loss import nppr_loss

def get_dataloader(data_path, numeric_cols, categorical_cols, time_col, max_past_events, batch_size, num_workers, train):
    """
    Prepares the data for the DataLoader.

    Args:
        data_path (str): Path to the Parquet file containing transactions.
        numeric_cols (list): List of column names for numeric features.
        categorical_cols (list): List of column names for categorical features.
        time_col (str): Column name for the time gap feature.
        max_past_events (int): Maximum number of past events to consider for time gaps.
        batch_size (int): The number of training samples processed together in a single pass.
        num_workers (int): Number of subprocesses to use for data loading.
        train (bool): Whether the dataset is a training or a testing one.

    Returns:
        DataLoader: DataLoader for the transaction dataset.
    """
    # Load the Parquet file into a DataFrame
    data = pd.read_parquet(data_path)

    # Category normalization
    for col in categorical_cols:
        unique_vals = sorted(data[col].unique())
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        data[col] = data[col].map(mapping)

    dataset = NPPRDataset(
        data=data,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        time_col=time_col,
        max_past_events=max_past_events
    )
    if train == True:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=dataset.collate_fn)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=dataset.collate_fn)

    return dataloader

class InferenceCallback(Callback):
    def __init__(self, checkpoint_callback, train_clf_dataloader, test_clf_dataloader, train_clf_path, test_clf_path,
                 train_target_path, embedding_dims, embedding_size, hidden_size_enc, hidden_size_dec,
                 num_numerical_features, num_categories, inference_device="cuda:1"):
        """
        Callback for inference and evaluation after each epoch.

        Args:
            checkpoint_callback: Reference to the ModelCheckpoint instance.
            train_clf_dataloader: Dataloader for training data.
            test_clf_dataloader: Dataloader for testing data.
            train_clf_path: Path to parquet file with training data.
            test_clf_path: Path to parquet file with testing data.
            train_target_path: Path to csv file with targets for app_ids.
            embedding_dims: Embedding dimensions for the model.
            embedding_size: Embedding size for the model.
            hidden_size_enc: Encoder hidden size.
            hidden_size_dec: Decoder hidden size.
            num_numerical_features: Number of numerical features.
            num_categories: List of number of categories for each categorical feature.
            inference_device: Device for inference (default is "cuda:1").
        """
        super().__init__()
        self.checkpoint_callback = checkpoint_callback
        self.train_clf_dataloader = train_clf_dataloader
        self.test_clf_dataloader = test_clf_dataloader
        self.train_clf = pd.read_parquet(train_clf_path)
        self.test_clf = pd.read_parquet(test_clf_path)
        self.train_target = pd.read_csv(train_target_path)
        self.embedding_dims = embedding_dims
        self.embedding_size = embedding_size
        self.hidden_size_enc = hidden_size_enc
        self.hidden_size_dec = hidden_size_dec
        self.num_numerical_features = num_numerical_features
        self.num_categories = num_categories
        self.inference_device = inference_device
        self.results = []

    def on_train_epoch_end(self, trainer, pl_module):
        logging.info("InferenceCallback: Starting on_train_epoch_end.")
        checkpoint_path = self.checkpoint_callback.last_model_path

        # Ensure checkpoint_path is valid
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            logging.warning(f"No checkpoint found for epoch {trainer.current_epoch}. Skipping inference.")
            return

        try:
            # Load the model for inference
            inference_module = NpprInferenceModule(
                model_path=checkpoint_path,
                embedding_dims=self.embedding_dims,
                embedding_size=self.embedding_size,
                hidden_size_enc=self.hidden_size_enc,
                hidden_size_dec=self.hidden_size_dec,
                num_numerical_features=self.num_numerical_features,
                num_categories=self.num_categories,
                device=torch.device(self.inference_device)
            )
            logging.info("Inference module initialized successfully.")
        except Exception as e:
            logging.error(f"Failed to initialize inference module: {e}")
            return

        # Perform inference and evaluation
        try:
            # Generate embeddings
            train_embeddings = inference_module.encode(self.train_clf_dataloader)
            test_embeddings = inference_module.encode(self.test_clf_dataloader)

            # Ensure embeddings are paired with app_id
            train_app_ids = self.train_clf['app_id'].unique()
            test_app_ids = self.test_clf['app_id'].unique()

            train_embeddings_df = pd.DataFrame(train_embeddings)
            train_embeddings_df['app_id'] = train_app_ids

            test_embeddings_df = pd.DataFrame(test_embeddings)
            test_embeddings_df['app_id'] = test_app_ids

            # Merge embeddings with targets
            train_data = train_embeddings_df.merge(self.train_target[['app_id', 'flag']], on='app_id', how='inner')
            test_data = test_embeddings_df.merge(self.train_target[['app_id', 'flag']], on='app_id', how='inner')

            # Extract features and targets
            X_train = train_data.drop(columns=['app_id', 'flag']).values
            y_train = train_data['flag'].values

            X_test = test_data.drop(columns=['app_id', 'flag']).values
            y_test = test_data['flag'].values

            # Train LGBM classifier and evaluate
            model = LGBMClassifier(
                random_state=42, n_estimators=500, boosting_type='gbdt', objective='binary',
                subsample=0.5, subsample_freq=1, learning_rate=0.02, max_depth=6, reg_alpha=1, reg_lambda=1,
                min_child_samples=50, colsample_bytree=0.75
            )
            model.fit(X_train, y_train)
            y_pred = model.predict_proba(X_test)[:, 1]
            roc_auc = roc_auc_score(y_test, y_pred)

            # Log and save results
            self.results.append({'Num Epoch': trainer.current_epoch + 1, 'Score': roc_auc})
            with open("results.txt", "a") as f:
                f.write(f"{trainer.current_epoch + 1}\t{roc_auc:.4f}\n")
            logging.info(f"Epoch {trainer.current_epoch + 1}: ROC AUC = {roc_auc:.4f}")
        except Exception as e:
            logging.error(f"Error during inference or evaluation: {e}")
