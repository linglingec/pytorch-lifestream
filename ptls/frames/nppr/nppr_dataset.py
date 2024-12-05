import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pandas as pd

class NPPRDataset(Dataset):
    def __init__(self, data: pd.DataFrame, numeric_cols=None, categorical_cols=None, time_col="hour_diff", 
                max_past_events=50, last_n_transactions=None, train=True):
        """
        Initializes the dataset for transaction sequences from a DataFrame.

        Args:
            data (pd.DataFrame): DataFrame containing transaction data with the specified columns.
            numeric_cols (list of str): List of column names for numeric features.
            categorical_cols (list of str): List of column names for categorical features.
            time_col (str): Column name representing time difference between transactions (for PR task).
            max_past_events (int): Maximum number of past events to consider for time gaps.
            last_n_transactions (int): Number of latest transactions to use during training. None for inference.
            train (bool): Flag to indicate whether the dataset is for training (True) or inference (False).
        """
        self.numeric_cols = numeric_cols if numeric_cols is not None else []
        self.categorical_cols = categorical_cols if categorical_cols is not None else []
        self.time_col = time_col
        self.max_past_events = max_past_events
        self.last_n_transactions = last_n_transactions
        self.train = train

        # Sort data by application ID and reset index
        data = data.sort_values(by=["app_id", "transaction_number"]).reset_index(drop=True)

        # Group by 'app_id' to define sequences
        self.sequences = [group for _, group in data.groupby("app_id")]
        self.data = data

    def __len__(self):
        # Number of sequences in the data
        return len(self.sequences)

    def __getitem__(self, index):
        """
        Returns a sequence of transactions for a given application ID.

        Args:
            index (int): Index of the sequence to retrieve.

        Returns:
            Tuple[Tensor, Tensor, list of list of float]: A tuple containing:
                - Numeric features tensor for the sequence
                - Categorical features tensor for the sequence
                - List of time gaps for each event up to max_past_events
        """
        sequence = self.sequences[index]

        # Select last N transactions for training, or all transactions for inference
        if self.train and self.last_n_transactions is not None:
            sequence = sequence.iloc[-self.last_n_transactions:]
        
        # Extract numeric features
        numeric_features = torch.tensor(sequence[self.numeric_cols].fillna(0).values, dtype=torch.float32)
        
        # Extract categorical features
        categorical_features = torch.tensor(sequence[self.categorical_cols].fillna(0).values, dtype=torch.long)
        
        # Calculate time gaps for each event
        hour_diffs = sequence[self.time_col].fillna(0).values
        hour_diffs[0] = -1
        time_gaps = []

        for t in range(len(hour_diffs)):
            if hour_diffs[t] == -1:  # First event in sequence
                time_gaps.append([])
            else:
                gaps = []
                for k in range(1, self.max_past_events + 1):
                    if t - k >= 0:
                        gaps.append(sum(hour_diffs[t - k + 1:t + 1]))
                time_gaps.append(gaps)

        return numeric_features, categorical_features, time_gaps

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle variable-length sequences by padding.

        Args:
            batch (list): A list of tuples from the Dataset.__getitem__.
                        Each tuple contains (numeric_features, categorical_features, time_gaps).

        Returns:
            Tuple of padded numeric features, padded categorical features, and padded time gaps.
        """
        numeric_features, categorical_features, time_gaps = zip(*batch)

        # Pad numeric features (dim 0 is batch, dim 1 is sequence length)
        padded_numeric_features = pad_sequence(numeric_features, batch_first=True, padding_value=0.0)

        # Pad categorical features
        padded_categorical_features = pad_sequence(categorical_features, batch_first=True, padding_value=0)

        # Handle time gaps: Pad with -1 to indicate missing gaps
        max_seq_len = max(len(tg) for tg in time_gaps)  # Maximum number of events in batch
        max_gap_len = max(len(gap_list) for tg in time_gaps for gap_list in tg)  # Maximum number of gaps per event

        # Create a 3D padded tensor for time gaps: (batch_size, max_seq_len, max_gap_len)
        padded_time_gaps = torch.full((len(batch), max_seq_len, max_gap_len), -1.0, dtype=torch.float32)

        for i, tg in enumerate(time_gaps):
            for j, gap_list in enumerate(tg):
                if len(gap_list) > 0:
                    gap_tensor = torch.tensor(gap_list, dtype=torch.float32)
                    padded_time_gaps[i, j, :len(gap_tensor)] = gap_tensor  # Fill up to the length of the gap list

        return padded_numeric_features, padded_categorical_features, padded_time_gaps