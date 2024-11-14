import torch
import torch.nn as nn
import torch.nn.functional as F

class TransactionEncoder(nn.Module):
    def __init__(self, num_numeric_features, embedding_dims, hidden_size=512, embedding_size=512):
        """
        Initializes the transaction encoder.

        Args:
            num_numeric_features (int): Number of numeric features.
            embedding_dims (dict): Dictionary with structure {feature_name: {'in': dictionary_size, 'out': embedding_size}} 
                                   for each categorical feature.
            hidden_size (int): Size of the hidden layers in the MLP and GRU.
            embedding_size (int): Output size of the transaction embedding.
        """
        super(TransactionEncoder, self).__init__()
        
        # Normalization layer for numeric features
        self.numeric_norm = nn.BatchNorm1d(num_numeric_features)
        
        # Initialize embedding layers for each categorical feature
        self.embeddings = nn.ModuleDict({
            feature: nn.Embedding(num_embeddings=params['in'], embedding_dim=params['out'])
            for feature, params in embedding_dims.items()
        })
        
        # Calculate total embedding dimension as the sum of all individual embedding dimensions
        total_embed_dim = sum(params['out'] for params in embedding_dims.values())
        
        # MLP for preprocessing features before GRU
        self.preprocess_mlp = nn.Sequential(
            nn.Linear(num_numeric_features + total_embed_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # GRU layer to process sequential data
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        
        # Projection layer to produce final transaction embedding
        self.proj_layer = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.Sigmoid()
        )

    def forward(self, x_numeric, x_categorical):
        """
        Forward pass for encoding a sequence of transactions.

        Args:
            x_numeric (Tensor): Tensor of numeric features with shape (batch_size, num_numeric_features).
            x_categorical (dict of Tensor): Dictionary of tensors for categorical features, with shape 
                                            (batch_size,) for each feature.

        Returns:
            e_t (Tensor): Embedding of the transaction with shape (batch_size, embedding_size).
            h_t (Tensor): Updated hidden state from the GRU with shape (1, batch_size, hidden_size).
        """
        # Normalize numeric features
        x_numeric = self.numeric_norm(x_numeric)
        
        # Pass each categorical feature through its respective embedding layer
        embedded_categorical_features = []
        for feature, emb_layer in self.embeddings.items():
            embedded_feature = emb_layer(x_categorical[feature])
            embedded_categorical_features.append(embedded_feature)
        
        # Concatenate all embedded categorical features along with numeric features
        x_embed = torch.cat(embedded_categorical_features, dim=-1)
        x = torch.cat([x_numeric, x_embed], dim=-1)
        
        # Pass through MLP
        x = self.preprocess_mlp(x).unsqueeze(1)  # Unsqueeze to add sequence dimension for GRU
        
        # Pass through GRU
        _, h_t = self.gru(x)
        
        # Project to final embedding space
        e_t = self.proj_layer(h_t[-1])  # Use last hidden state for embedding
        
        return e_t, h_t

# Decoder for NP task
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
            e_t (Tensor): Event embedding for NP task.
        
        Returns:
            Tuple[Tensor, list of Tensor]: Numerical outputs and categorical probabilities.
        """
        x = self.decoder(e_t)
        output = self.output_layer(x)
        
        # Split into numerical and categorical outputs
        numerical_outputs = output[:, :self.num_numerical_features]
        
        categorical_outputs = []
        start_idx = self.num_numerical_features
        for dim in self.num_categories:
            categorical_part = output[:, start_idx:start_idx + dim]
            categorical_outputs.append(torch.softmax(categorical_part, dim=-1))
            start_idx += dim
        
        return numerical_outputs, categorical_outputs

# Decoder for PR task
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
        
        # MLP decoder layers for PR task, takes time gap as an additional input
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size + 1, hidden_size),  # +1 for the time gap input
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )
        
        # Output layer for generating outputs for numerical and categorical features
        self.output_layer = nn.Linear(hidden_size, num_numerical_features + sum(num_categories))

    def forward(self, e_t, time_gap):
        """
        Forward pass for the PR decoder.

        Args:
            e_t (Tensor): Event embedding for PR task.
            time_gap (Tensor): Time difference between current and past events.
        
        Returns:
            Tuple[Tensor, list of Tensor]: Numerical outputs and categorical probabilities.
        """
        # Concatenate embedding with time gap
        pr_input = torch.cat([e_t, time_gap.unsqueeze(1)], dim=-1)
        
        # Pass through PR decoder MLP
        x = self.decoder(pr_input)
        output = self.output_layer(x)
        
        # Split into numerical and categorical outputs
        numerical_outputs = output[:, :self.num_numerical_features]
        
        categorical_outputs = []
        start_idx = self.num_numerical_features
        for dim in self.num_categories:
            categorical_part = output[:, start_idx:start_idx + dim]
            categorical_outputs.append(torch.softmax(categorical_part, dim=-1))
            start_idx += dim
        
        return numerical_outputs, categorical_outputs

# Main model combining encoder and two decoders
class NPPRModel(nn.Module):
    def __init__(self, embedding_dims, embedding_size=512, hidden_size=512, num_numerical_features=1, num_categories=[]):
        """
        Initializes the NPPR model.

        Args:
            embedding_dims (dict): Dictionary with structure {feature_name: {'in': dictionary_size, 'out': embedding_size}} 
                                   for each categorical feature.
            num_numerical_features (int): Number of numeric features.
            hidden_size (int): Hidden size for the encoder and decoders.
            embedding_size (int): Size of the final embedding.
            num_categories (list of int): List where each element represents the number of unique categories 
                                          for a categorical feature.
        """
        super(NPPRModel, self).__init__()
        self.encoder = TransactionEncoder(num_numeric_features, embedding_dims, hidden_size, embedding_size)
        self.np_decoder = NPDecoder(embedding_size, hidden_size, num_numerical_features, num_categories)
        self.pr_decoder = PRDecoder(embedding_size, hidden_size, num_numerical_features, num_categories)

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