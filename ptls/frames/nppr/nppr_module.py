import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import pytorch_lightning as pl
from nppr_loss import nppr_loss

class NPPREncoder(nn.Module):
    def __init__(self, num_numerical_features, embedding_dims, hidden_size, embedding_size):
        """
        Initializes the NPPREncoder.

        Args:
            num_numerical_features (int): Number of numerical features.
            embedding_dims (dict): Dictionary of embedding sizes for categorical features.
                                   Example: {"currency": {"in": 5, "out": 16}, ...}.
            hidden_size (int): Hidden size for the GRU layer.
            embedding_size (int): Final embedding size.
        """
        super(NPPREncoder, self).__init__()

        self.embedding_dims = embedding_dims

        # BatchNorm for numerical features
        self.numeric_norm = nn.BatchNorm1d(num_numerical_features)

        # Embedding layers for categorical features
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(num_embeddings=dims["in"], embedding_dim=dims["out"], padding_idx=-1)
            for feature, dims in embedding_dims.items()
        })

        # Total input size for GRU
        input_size = num_numerical_features + sum(dims["out"] for dims in embedding_dims.values())

        # MLP with two hidden layers
        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # GRU layer
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)

        # Final projection layer
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.Sigmoid()
        )

    def forward(self, x_numeric, x_categorical):
        """
        Forward pass for the NPPREncoder.

        Args:
            x_numeric (Tensor): Numerical features, shape (batch_size, max_seq_len, num_numerical_features).
            x_categorical (Tensor): Categorical features, shape (batch_size, max_seq_len, num_categorical_features).

        Returns:
            Tuple[Tensor, Tensor]: Event embeddings (e_t) and hidden states (h_t) from the GRU.
        """
        batch_size, max_seq_len, _ = x_numeric.size()

        # Reshape for BatchNorm1d: Flatten sequence for normalization
        x_numeric_flat = x_numeric.view(-1, x_numeric.size(-1))  # Shape: (batch_size * max_seq_len, num_numerical_features)
        x_numeric_normalized = self.numeric_norm(x_numeric_flat)
        x_numeric_normalized = x_numeric_normalized.view(batch_size, max_seq_len, -1)

        # Process categorical features: Apply embedding and concatenate
        x_categorical_embedded = []
        for i, (feature_name, embedding_layer) in enumerate(self.embeddings.items()):
            feature_values = x_categorical[:, :, i]
            embedded = embedding_layer(feature_values)  # Shape: (batch_size, max_seq_len, embedding_dim)
            x_categorical_embedded.append(embedded)

        x_categorical_combined = torch.cat(x_categorical_embedded, dim=-1)  # Shape: (batch_size, max_seq_len, total_embedding_dim)

        # Concatenate numerical and categorical features
        x_combined = torch.cat([x_numeric_normalized, x_categorical_combined], dim=-1)  # Shape: (batch_size, max_seq_len, input_size)

        # Pass through MLP
        x_mlp = self.mlp(x_combined)  # Shape: (batch_size, max_seq_len, hidden_size)

        # Pass through GRU
        h_t, _ = self.gru(x_mlp) # h_t: All hidden states, shape (batch_size, max_seq_len, hidden_size)

        # Project to embedding space
        e_t = self.projection(h_t)  # Shape: (batch_size, max_seq_len, embedding_size)

        return e_t, h_t

class NPDecoder(nn.Module):
    def __init__(self, embedding_size=512, hidden_size=512, num_numerical_features=1, num_categories=[]):
        """
        Initializes the NP decoder.

        Args:
            embedding_size (int): Size of the embedding vector.
            hidden_size (int): Hidden layer size for the MLP.
            num_numerical_features (int): Number of numerical features to decode.
            num_categories (list of int): List where each element represents the number of unique categories 
                                          for a categorical feature.
        """
        super(NPDecoder, self).__init__()
        
        self.num_numerical_features = num_numerical_features
        self.num_categories = num_categories

        # MLP decoder layers for NP task
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Output layer for generating outputs for numerical and categorical features
        self.output_layer = nn.Linear(hidden_size, num_numerical_features + sum(num_categories))

    def forward(self, e_t):
        """
        Forward pass for the NP decoder.

        Args:
            e_t (Tensor): Event embedding for NP task, shape (batch_size, seq_len, embedding_size).
        
        Returns:
            Tuple[Tensor, list of Tensor]: Numerical outputs and categorical logits.
        """
        batch_size, seq_len, embedding_size = e_t.shape

        # Flatten sequence for processing
        e_t_flat = e_t.view(batch_size * seq_len, embedding_size)

        # Pass through the decoder MLP
        x = self.decoder(e_t_flat)
        output = self.output_layer(x)

        # Split outputs into numerical and categorical components
        numerical_outputs = output[:, :self.num_numerical_features]
        numerical_outputs = numerical_outputs.view(batch_size, seq_len, -1)

        categorical_outputs = []
        start_idx = self.num_numerical_features
        for dim in self.num_categories:
            cat_output = output[:, start_idx:start_idx + dim]
            categorical_outputs.append(cat_output.view(batch_size, seq_len, dim))
            start_idx += dim

        return numerical_outputs, categorical_outputs

class PRDecoder(nn.Module):
    def __init__(self, embedding_size=512, hidden_size=512, num_numerical_features=1, num_categories=[]):
        """
        Initializes the PR decoder.

        Args:
            embedding_size (int): Size of the embedding vector.
            hidden_size (int): Hidden layer size for the MLP.
            num_numerical_features (int): Number of numerical features to decode.
            num_categories (list of int): List where each element represents the number of unique categories 
                                          for a categorical feature.
        """
        super(PRDecoder, self).__init__()
        
        self.num_numerical_features = num_numerical_features
        self.num_categories = num_categories
        
        # MLP decoder layers for PR task
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size + 1, hidden_size),  # +1 for the concatenated time gap
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Output layer for generating outputs for numerical and categorical features
        self.output_layer = nn.Linear(hidden_size, num_numerical_features + sum(num_categories))

    def forward(self, e_t, time_gaps):
        """
        Forward pass for the PR decoder.

        Args:
            e_t (Tensor): Event embedding for PR task, shape (batch_size, seq_len, embedding_size).
            time_gaps (Tensor): Time gaps tensor, shape (batch_size, seq_len, max_gap_len), with -1 indicating padding.

        Returns:
            Tuple[Tensor, list of Tensor]: Numerical outputs and categorical logits for each event.
        """
        batch_size, seq_len, max_gap_len = time_gaps.shape

        # Mask for valid time gaps
        valid_time_mask = (time_gaps != -1)

        # Use valid time gaps directly in concatenation
        time_gaps_masked = torch.where(valid_time_mask, time_gaps, torch.tensor(0.0, device=time_gaps.device))

        # Expand embeddings for max_gap_len
        e_t_expanded = e_t.unsqueeze(2).expand(-1, -1, max_gap_len, -1)  # Shape: (batch_size, seq_len, max_gap_len, embedding_size)

        # Concatenate embeddings with time gaps
        pr_inputs = torch.cat([e_t_expanded, time_gaps_masked.unsqueeze(-1)], dim=-1)  # Shape: (batch_size, seq_len, max_gap_len, embedding_size + 1)

        # Flatten sequence for processing
        pr_inputs_flat = pr_inputs.view(batch_size * seq_len * max_gap_len, -1)  # Shape: (batch_size * seq_len * max_gap_len, embedding_size + 1)

        # Pass through the decoder MLP
        x = self.decoder(pr_inputs_flat)
        output = self.output_layer(x)

        # Split outputs into numerical and categorical components
        numerical_outputs = output[:, :self.num_numerical_features]
        numerical_outputs = numerical_outputs.view(batch_size, seq_len, max_gap_len, -1)  # Shape: (batch_size, seq_len, max_gap_len, num_numerical_features)

        categorical_outputs = []
        start_idx = self.num_numerical_features
        for dim in self.num_categories:
            cat_output = output[:, start_idx:start_idx + dim]
            categorical_outputs.append(cat_output.view(batch_size, seq_len, max_gap_len, dim))
            start_idx += dim

        return numerical_outputs, categorical_outputs

# Main model combining encoder and two decoders
class NpprPretrainModule(pl.LightningModule):
    def __init__(self, 
                embedding_dims, 
                embedding_size=512,
                hidden_size_enc=512, 
                hidden_size_dec=512, 
                num_numerical_features=1, 
                num_categories=[],
                learning_rate=1e-3,
                total_steps=1000,
                pct_start=0.3,
                lambda_param=1.0,
                alpha=1.0):
        """
        Initializes the NPPR model.

        Args:
            embedding_dims (dict): Dictionary with structure {feature_name: {'in': dictionary_size, 'out': embedding_size}} 
                                   for each categorical feature.
            num_numerical_features (int): Number of numeric features.
            hidden_size_enc (int): Hidden size for the encoder.
            hidden_size_dec (int): Hidden size for the decoders.
            embedding_size (int): Size of the final embedding.
            num_categories (list of int): List where each element represents the number of unique categories 
                                          for a categorical feature.
            learning_rate (float): Learning rate for the optimizer.
            total_steps (int): Total steps for the OneCycleLR scheduler.
            pct_start (float): Percentage of the cycle spent increasing the learning rate.
            lambda_param (float): Decay parameter for PR loss.
            alpha (float): Weight for combining NP and PR losses.
        """
        super().__init__()
        self.save_hyperparameters()
        self.encoder = NPPREncoder(num_numerical_features, embedding_dims, hidden_size_enc, embedding_size)
        self.np_decoder = NPDecoder(embedding_size, hidden_size_dec, num_numerical_features, num_categories)
        self.pr_decoder = PRDecoder(embedding_size, hidden_size_dec, num_numerical_features, num_categories)

        self.learning_rate = learning_rate
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.lambda_param = lambda_param
        self.alpha = alpha

    def forward(self, x_numeric, x_categorical, time_gaps):
        """
        Forward pass for the NPPR model.

        Args:
            x_numeric (Tensor): Numeric feature tensor.
            x_categorical (Tensor): Categorical feature tensor.
            time_gaps (Tensor): Time gaps between events.
        
        Returns:
            Tuple containing embeddings and outputs for NP and PR tasks.
        """
        e_t, h_t = self.encoder(x_numeric, x_categorical)
        np_numerical, np_categorical = self.np_decoder(e_t)
        pr_numerical, pr_categorical = self.pr_decoder(e_t, time_gaps)
        
        return e_t, np_numerical, np_categorical, pr_numerical, pr_categorical, h_t
    
    def training_step(self, batch, batch_idx):
        """
        Training step for PyTorch Lightning.

        Args:
            batch (tuple): A batch of input data containing numeric, categorical, and time gap tensors.
            batch_idx (int): Index of the current batch.

        Returns:
            Tensor: Computed loss for the current batch.
        """
        x_numeric, x_categorical, time_gaps = batch

        # Prepare targets for NP and PR tasks
        target_np_numerical = x_numeric[:, 1:, :]
        target_np_categorical = [x_categorical[:, 1:, i] for i in range(x_categorical.size(-1))]
        target_pr_numerical = x_numeric[:, :-1].unsqueeze(2).expand(-1, -1, time_gaps.size(-1), -1)
        target_pr_categorical = [
            x_categorical[:, :-1, i].unsqueeze(2).repeat(1, 1, time_gaps.size(-1))
            for i in range(x_categorical.size(-1))
        ]

        # Forward pass
        e_t, np_numerical, np_categorical, pr_numerical, pr_categorical, _ = self.forward(
            x_numeric, x_categorical, time_gaps
        )
        np_numerical = np_numerical[:, :-1, :]
        np_categorical = [pred[:, :-1, :] for pred in np_categorical]
        pr_numerical = pr_numerical[:, 1:, :, :]
        pr_categorical = [pred[:, 1:, :, :] for pred in pr_categorical]
        time_gaps = time_gaps[:, :-1, :]

        # Compute loss
        loss = nppr_loss(
            np_numerical, np_categorical, pr_numerical, pr_categorical,
            target_np_numerical, target_np_categorical,
            target_pr_numerical, target_pr_categorical,
            time_gaps, self.lambda_param, self.alpha
        )
        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        """
        Validation step for a single batch.

        Args:
            batch: Batch of data from the dataloader.
            batch_idx: Index of the batch.

        Returns:
            torch.Tensor: Validation loss for the batch.
        """
        x_numeric, x_categorical, time_gaps = batch
        target_np_numerical = x_numeric[:, 1:, :]
        target_np_categorical = [x_categorical[:, 1:, i] for i in range(x_categorical.size(-1))]
        target_pr_numerical = x_numeric[:, :-1].unsqueeze(2).expand(-1, -1, time_gaps.size(-1), -1)
        target_pr_categorical = [
            x_categorical[:, :-1, i].unsqueeze(2).repeat(1, 1, time_gaps.size(-1))
            for i in range(x_categorical.size(-1))
        ]
        e_t, np_numerical, np_categorical, pr_numerical, pr_categorical, _ = self(
            x_numeric, x_categorical, time_gaps
        )
        np_numerical = np_numerical[:, :-1, :]
        np_categorical = [pred[:, :-1, :] for pred in np_categorical]
        pr_numerical = pr_numerical[:, 1:, :, :]
        pr_categorical = [pred[:, 1:, :, :] for pred in pr_categorical]
        time_gaps = time_gaps[:, :-1, :]
        loss = nppr_loss(
            np_numerical, np_categorical, pr_numerical, pr_categorical,
            target_np_numerical, target_np_categorical,
            target_pr_numerical, target_pr_categorical,
            time_gaps, self.lambda_param, self.alpha
        )
        self.log("val_loss", loss)
        return loss
    
    def configure_optimizers(self):
        """
        Configures the optimizer and learning rate scheduler.

        Returns:
            Configuration for optimizer and scheduler.
        """
        optim = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optim,
            max_lr=self.learning_rate,
            total_steps=self.total_steps,
            pct_start=self.pct_start,
            anneal_strategy='cos',
            cycle_momentum=False,
            div_factor=25.0,
            final_div_factor=10000.0,
            three_phase=False
        )
        scheduler = {'scheduler': scheduler, 'interval': 'step', 'frequency': 1}

        return [optim], [scheduler]

class NpprInferenceModule:
    """
    Inference module for NpprPretrainModule to generate embeddings or predictions.

    Args:
        model_path (str): Path to the pretrained model weights.
        embedding_dims (dict): Same embedding dimensions used during training.
        embedding_size (int): Embedding size for the encoder.
        hidden_size_enc (int): Hidden size for the encoder GRU.
        hidden_size_dec (int): Hidden size for the decoders.
        num_numerical_features (int): Number of numerical features.
        num_categories (list): List containing the number of unique categories for each categorical feature.
        device (torch.device or str): Device to use for inference ('cuda', 'cpu', or None for auto-detection).
    """
    def __init__(self, model_path, embedding_dims, embedding_size=512, hidden_size_enc=512,
                 hidden_size_dec=512, num_numerical_features=1, num_categories=[], device=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.model = NpprPretrainModule(
            embedding_dims=embedding_dims,
            embedding_size=embedding_size,
            hidden_size_enc=hidden_size_enc,
            hidden_size_dec=hidden_size_dec,
            num_numerical_features=num_numerical_features,
            num_categories=num_categories
        )
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["state_dict"])
        self.model.to(self.device)
        self.model.eval()

    def encode(self, dataloader: DataLoader):
        """
        Generates embeddings for the entire dataset using the encoder.

        Args:
            dataloader (DataLoader): Dataloader providing batches of input data.

        Returns:
            torch.Tensor: Sequence embeddings for the entire dataset.
        """
        sequence_embeddings = []

        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Encoding"):
                x_numeric, x_categorical, _ = batch
                x_numeric = x_numeric.to(self.device, non_blocking=True)
                x_categorical = x_categorical.to(self.device, non_blocking=True)

                # Pass through encoder
                e_t, _ = self.model.encoder(x_numeric, x_categorical)

                # Average embeddings across sequence length
                sequence_embedding = e_t.mean(dim=1)
                sequence_embeddings.append(sequence_embedding.cpu())

            sequence_embeddings = torch.cat(sequence_embeddings, dim=0).detach().numpy()

        return sequence_embeddings
