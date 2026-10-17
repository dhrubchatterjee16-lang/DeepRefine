import torch
import torch.nn as nn

class RefineryPredictiveMaintenanceLSTM(nn.Module):
    """
    Bi-directional LSTM with multi-task heads for refinery failure & RUL prediction.
    Pure PyTorch implementation (zero external framework dependency).
    """
    def __init__(
        self,
        n_features: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.25
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0
        )

        combined_dim = hidden_dim * 2

        # Classification head: Failure probability logits
        self.classifier = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

        # Regression head: RUL in hours
        self.regressor = nn.Sequential(
            nn.Linear(combined_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        # x shape: [batch_size, seq_len, n_features]
        lstm_out, _ = self.lstm(x)
        last_hidden = lstm_out[:, -1, :]
        prob_logits = self.classifier(last_hidden).squeeze(-1)
        rul_estimate = self.regressor(last_hidden).squeeze(-1)
        return prob_logits, rul_estimate
