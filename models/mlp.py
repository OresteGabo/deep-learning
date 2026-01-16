import torch.nn as nn

class EarthquakeMLP(nn.Module):
    """
    Multi-Layer Perceptron (Baseline) for seismic classification.

    This model treats the input time series as a flat vector of independent features.
    It serves as a performance floor to compare against more complex architectures.
    """
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
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Raw seismic signal of shape (batch_size, 512).

        Returns:
            torch.Tensor: Unnormalized class logits of shape (batch_size, 2).
        """
        return self.net(x)