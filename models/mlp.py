import torch.nn as nn

class EarthquakeMLP(nn.Module):
    def __init__(self, input_size=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )
    def forward(self, x): return self.net(x)