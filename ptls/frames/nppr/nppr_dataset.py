import torch
from torch.utils.data import Dataset
import pandas as pd

class TransactionDataset(Dataset):
    def __init__(self, data: pd.DataFrame, sequence_length=10, numeric_cols=None, categorical_cols=None, time_col="hour_diff"):
        """
        Initializes the dataset for transaction sequences from a DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing transaction data with the specified columns.
            sequence_length (int): Length of the sequence (window) of events.
            numeric_cols (list of str): List of column names for numeric features.
            categorical_cols (list of str): List of column names for categorical features.
            time_col (str): Column name representing time difference between transactions (for PR task).
        """
        self.sequence_length = sequence_length
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.time_col = time_col

        # Sort the data by the unique identifier to ensure correct time sequence
        data = data.sort_values(by=["app_id"]).reset_index(drop=True)
        self.data = data

    def __len__(self):
        # Number of available sequences in the data
        return len(self.data) - self.sequence_length

    def __getitem__(self, index):
        """
        Returns a sequence of transactions starting from the given index.

        Args:
            index (int): Index of the first event in the sequence.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: A tuple containing:
                - Numeric features tensor for the sequence
                - Categorical features tensor for the sequence
                - Time differences tensor for each transaction in the sequence
        """
        # Extract the sequence of transactions
        sequence = self.data.iloc[index:index + self.sequence_length]
        
        # Extract numeric and categorical features, filling missing values
        numeric_features = torch.tensor(sequence[self.numeric_cols].fillna(0).values, dtype=torch.float32)
        categorical_features = torch.tensor(sequence[self.categorical_cols].fillna(0).values, dtype=torch.long)
        
        # Time difference values for PR task
        time_gaps = torch.tensor(sequence[self.time_col].fillna(0).values, dtype=torch.float32)

        return numeric_features, categorical_features, time_gaps