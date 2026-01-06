
import torch.nn as nn


class EarthquakeCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # First block: looks for small ripples
            nn.Conv1d(1, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),  # Helps stabilize training on M4
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Second block: looks for larger patterns
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            # Third block: complex wave combinations
            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)  # Reduces everything to a single value per filter
        )
        self.fc = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),  # Regularization (10% of grade!)
            nn.Linear(64, 2)
        )

    def forward(self, x):
        x = x.unsqueeze(1)  # From [batch, 512] to [batch, 1, 512]
        x = self.conv(x).squeeze(-1)
        return self.fc(x)