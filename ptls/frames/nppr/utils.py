import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd
from tqdm import tqdm
from nppr_dataset import NPPRDataset
from nppr_loss import nppr_loss

def get_dataloader(data, numeric_cols, categorical_cols, time_col, max_past_events, batch_size, num_workers, train):
    """
    Prepares the data for the DataLoader.

    Args:
        df (pd.DataFrame): Input DataFrame containing transactions.
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

def train(model, dataloader, optimizer, epochs=10, lambda_param=1.0, alpha=1.0, save_path="nppr_model.pth", device=None):
    """
    Trains the NPPRModel using the provided dataloader.

    Args:
        model (NPPRModel): The NPPR model to train.
        dataloader (DataLoader): Dataloader providing batches of data.
        optimizer (torch.optim.Optimizer): Optimizer for training.
        epochs (int): Number of epochs.
        lambda_param (float): Decay parameter for PR loss.
        alpha (float): Weight for combining NP and PR losses.
        save_path (str): Path to save the trained model weights.
        device (torch.device or str): Device to use for encoding ('cuda', 'cpu', or None)

    Returns:
        NPPRModel: The trained model.
    """
     # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)
    
    model = model.to(device, non_blocking=True)

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}"):
            # Unpack batch
            x_numeric, x_categorical, time_gaps = batch

            # Move data to device
            x_numeric = x_numeric.to(device, non_blocking=True)
            x_categorical = x_categorical.to(device, non_blocking=True)
            time_gaps = time_gaps.to(device, non_blocking=True)

            # Prepare targets for NP and PR tasks
            # Target for NP task: Next event in sequence
            target_np_numerical = x_numeric[:, 1:, :]  # Shifted sequence
            target_np_categorical = [x_categorical[:, 1:, i] for i in range(x_categorical.size(-1))]

            # Target for PR task: Previous events in sequence
            target_pr_numerical = x_numeric[:, :-1].unsqueeze(2).expand(-1, -1, time_gaps.size(-1), -1)
            target_pr_categorical = [
                x_categorical[:, :-1, i].unsqueeze(2).repeat(1, 1, time_gaps.size(-1))
                for i in range(x_categorical.size(-1))
            ]

            # Forward pass
            e_t, np_numerical, np_categorical, pr_numerical, pr_categorical, _ = model(x_numeric, x_categorical, time_gaps)
            np_numerical = np_numerical[:, :-1, :]   # Align outputs and targets
            np_categorical = [pred[:, :-1, :] for pred in np_categorical]
            pr_numerical = pr_numerical[:, 1:, :, :]
            pr_categorical = [pred[:, 1:, :, :] for pred in pr_categorical]
            time_gaps = time_gaps[:, :-1, :]

            # Compute loss
            loss = nppr_loss(np_numerical, np_categorical, pr_numerical, pr_categorical, target_np_numerical, target_np_categorical, 
                    target_pr_numerical, target_pr_categorical, time_gaps, lambda_param, alpha)

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch + 1}: Loss = {epoch_loss / len(dataloader):.4f}")

    # Save model weights
    torch.save(model.state_dict(), save_path)
    print(f"Model saved to {save_path}")

    return model

def encode(model, dataloader, load_path="nppr_model.pth", device=None):
    """
    Encodes the transaction sequences into a single embedding per sequence by averaging the event embeddings.

    Args:
        model (NPPRModel): The NPPR model for encoding.
        dataloader (DataLoader): Dataloader providing batches of data.
        load_path (str): Path to load the trained model weights.
        device (torch.device or str): Device to use for encoding ('cuda', 'cpu', or None for auto-detection).

    Returns:
        Tensor: A tensor containing a single embedding for each sequence in the dataset.
    """
    # Determine device if not provided
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    # Load model weights
    model.load_state_dict(torch.load(load_path, map_location=device))
    model = model.to(device, non_blocking=True)
    model.eval()

    sequence_embeddings = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Encoding"):
            # Unpack batch
            x_numeric, x_categorical, _ = batch

            # Move data to the specified device
            x_numeric = x_numeric.to(device, non_blocking=True)
            x_categorical = x_categorical.to(device, non_blocking=True)

            # Forward pass through encoder
            e_t, _ = model.encoder(x_numeric, x_categorical)  # Shape: (batch_size, max_seq_len, embedding_size)

            # Average event embeddings along the sequence length dimension
            sequence_embedding = e_t.mean(dim=1)  # Shape: (batch_size, embedding_size)
            sequence_embeddings.append(sequence_embedding.cpu())

    # Concatenate all sequence embeddings
    sequence_embeddings = torch.cat(sequence_embeddings, dim=0)  # Shape: (total_sequences, embedding_size)

    return sequence_embeddings.cpu().detach().numpy()