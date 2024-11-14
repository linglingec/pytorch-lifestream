import torch
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm

def prepare_data(df, sequence_length, numeric_cols, categorical_cols, time_col):
    """
    Prepares the data for the DataLoader.

    Args:
        df (pd.DataFrame): Input DataFrame containing transactions.
        sequence_length (int): Length of transaction sequence for each data point.
        numeric_cols (list): List of column names for numeric features.
        categorical_cols (list): List of column names for categorical features.
        time_col (str): Column name for the time gap feature.

    Returns:
        DataLoader: DataLoader for the transaction dataset.
    """
    dataset = TransactionDataset(
        data=df,
        sequence_length=sequence_length,
        numeric_cols=numeric_cols,
        categorical_cols=categorical_cols,
        time_col=time_col
    )
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    return dataloader

def train(model, dataloader, num_epochs, optimizer, lambda_param=1.0, alpha=0.5):
    """
    Trains the NPPR model.

    Args:
        model (nn.Module): NPPR model instance.
        dataloader (DataLoader): DataLoader with training data.
        num_epochs (int): Number of training epochs.
        optimizer (torch.optim.Optimizer): Optimizer for model training.
        lambda_param (float): Decay parameter for PR task.
        alpha (float): Weight factor for combining NP and PR losses.
    """
    model.train()
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            x_numeric, x_categorical, time_gaps = batch

            # Forward pass
            e_t, np_numerical, np_categorical, pr_numerical, pr_categorical, _ = model(x_numeric, x_categorical, time_gaps)

            # Compute loss
            np_loss_value = np_loss(np_numerical, np_categorical, x_numeric, x_categorical)
            pr_loss_value = pr_loss(pr_numerical, pr_categorical, x_numeric, x_categorical, time_gaps, lambda_param)
            loss = (1 - alpha) * np_loss_value + alpha * pr_loss_value

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss/len(dataloader):.4f}")

def encode(model, dataloader):
    """
    Runs inference on the NPPR model.

    Args:
        model (nn.Module): NPPR model instance.
        dataloader (DataLoader): DataLoader with inference data.

    Returns:
        list: List of embeddings for each data point.
    """
    model.eval()
    embeddings = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Inference"):
            x_numeric, x_categorical, time_gaps = batch

            # Forward pass through encoder to get embeddings
            e_t, _, _, _, _, _ = model(x_numeric, x_categorical, time_gaps)
            embeddings.append(e_t.cpu().numpy())
    
    return embeddings