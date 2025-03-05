import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout=0.1):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.norm = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm(x + residual)  # Residual connection
        return x

class TiDE(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, context_size, horizon, num_encoder_layers=2, num_decoder_layers=2, decoder_output_dim=16, temporal_decoder_hidden=64, dropout=0.1):
        super(TiDE, self).__init__()
        self.context_size = context_size
        self.horizon = horizon

        # Feature projection for dynamic covariates
        self.feature_projection = ResidualBlock(input_dim, hidden_dim, dropout)
        
        # Dense Encoder
        self.encoder = nn.Sequential(
            *[ResidualBlock(input_dim * context_size, hidden_dim, dropout) for _ in range(num_encoder_layers)]
        )
        
        # Dense Decoder
        self.decoder = nn.Sequential(
            *[ResidualBlock(hidden_dim, hidden_dim, dropout) for _ in range(num_decoder_layers)]
        )
        
        # Temporal Decoder (per time-step in horizon)
        self.temporal_decoder = ResidualBlock(hidden_dim + input_dim, temporal_decoder_hidden, dropout)
        self.output_layer = nn.Linear(temporal_decoder_hidden, 1)  # Linear highway to allow future covariates

        # Global residual connection
        self.global_residual = nn.Linear(context_size, horizon)
    
    def forward(self, x_past, x_future, covariates_past, covariates_future):
        batch_size = x_past.shape[0]
        
        # Project covariates
        covariates_past = self.feature_projection(covariates_past)
        covariates_future = self.feature_projection(covariates_future)
        
        # Flatten input and encode
        x_encoded = torch.cat([x_past.flatten(start_dim=1), covariates_past.flatten(start_dim=1)], dim=-1)
        encoding = self.encoder(x_encoded)
        
        # Decode into horizon embeddings
        decoding = self.decoder(encoding)
        decoding = decoding.view(batch_size, self.horizon, -1)
        
        # Temporal decoding with future covariates
        temp_decoded = self.temporal_decoder(torch.cat([decoding, covariates_future], dim=-1))
        y_pred = self.output_layer(temp_decoded).squeeze(-1)
        
        # Global residual connection
        y_pred += self.global_residual(x_past).squeeze(-1)
        
        return y_pred

# Example usage
input_dim = 10  # Feature size per time-step
hidden_dim = 64
output_dim = 1
context_size = 96  # Lookback window
horizon = 24  # Forecast window

model = TiDE(input_dim, hidden_dim, output_dim, context_size, horizon)

# Dummy input tensors
batch_size = 32
x_past = torch.randn(batch_size, context_size, input_dim)
x_future = torch.randn(batch_size, horizon, input_dim)
covariates_past = torch.randn(batch_size, context_size, input_dim)
covariates_future = torch.randn(batch_size, horizon, input_dim)

# Forward pass
y_pred = model(x_past, x_future, covariates_past, covariates_future)
print(y_pred.shape)  # Expected output: (batch_size, horizon)
