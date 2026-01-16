import torch.nn as nn

class EarthquakeRNN(nn.Module):
    """
    Recurrent Neural Network using LSTM cells for seismic time-series analysis.

    This architecture is designed to capture long-term temporal dependencies
    across the 512-hour window, which is critical for identifying earthquake precursors.
    """
    def __init__(self):
        super().__init__()
        # Stacked LSTM: processes the signal point-by-point to maintain context
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        # Classifier: uses the final hidden state to predict the category
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input signals of shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: Logits of shape (batch_size, 2).
        """
        # LSTM expects (batch, seq, features), so we add the feature dimension
        x = x.unsqueeze(-1)

        # hn[-1] contains the hidden state from the last LSTM layer after the full sequence
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])