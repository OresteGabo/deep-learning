import torch.nn as nn

class EarthquakeRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=64, num_layers=2, batch_first=True)
        self.fc = nn.Linear(64, 2)
    def forward(self, x):
        x = x.unsqueeze(-1)
        _, (hn, _) = self.lstm(x)
        return self.fc(hn[-1])