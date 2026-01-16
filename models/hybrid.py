import torch
import torch.nn as nn


class SeismicNet(nn.Module):
    """
    Hybrid CNN-LSTM architecture for real-time seismic monitoring.
    
    Combines 1D convolutions for local feature extraction (spatial) with a 
    Bi-Directional LSTM for sequential context (temporal). Optimized for 
    high-speed inference.
    """
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=2):
        super(SeismicNet, self).__init__()

        # 1. Spatial Feature Extractor (CNN)
        # Reduces the 512-length signal into a dense representation
        self.feature_extractor = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )

        # 2. Temporal Context (Bi-LSTM)
        # Bidirectional processing looks at both past and "future" context in the window
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)

        # 3. Classification Head
        # hidden_dim * 2 because the LSTM is bidirectional (concatenates both directions)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw signal of shape (batch, 512) or (batch, 1, 512).

        Returns:
            torch.Tensor: Prediction logits for 2 classes.
        """
        # Ensure input has channel dimension for Conv1d
        if x.dim() == 2: 
            x = x.unsqueeze(1)
            
        # Extract spatial features: [batch, 64, 128]
        x = self.feature_extractor(x)
        
        # Prepare for LSTM: Transpose from (batch, channels, seq) to (batch, seq, channels)
        x = x.transpose(1, 2)
        
        # lstm_out shape: [batch, seq, hidden_dim * 2]
        lstm_out, _ = self.lstm(x)
        
        # Take the final time-step as the representative sequence summary
        last_step = lstm_out[:, -1, :]
        
        return self.classifier(last_step)