import torch
import torch.nn as nn


class SeismicNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=64, num_classes=2):
        super(SeismicNet, self).__init__()

        # 1. Spatial Features (CNN)
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
        self.lstm = nn.LSTM(64, hidden_dim, num_layers=1,
                            batch_first=True, bidirectional=True)

        # 3. Classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = self.feature_extractor(x)
        x = x.transpose(1, 2)
        lstm_out, _ = self.lstm(x)
        last_step = lstm_out[:, -1, :]
        return self.classifier(last_step)