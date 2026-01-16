import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import torch
import numpy as np

class SeismicTransform:
    def __init__(self, mean=0.0, std=1.0, noise_level=0.01):
        self.mean = mean
        self.std = std
        self.noise_level = noise_level

    def __call__(self, x):
        # Scaling + Noise
        x = (x - self.mean) / (self.std + 1e-6)
        noise = torch.randn_like(x) * self.noise_level
        return x + noise

# Apply this inside your Dataset class or data loading loop
def get_dataloaders(train_path, test_path, batch_size=32):
    # Load TSV files [cite: 23]
    train_df = pd.read_csv(train_path, sep='\t', header=None)
    test_df = pd.read_csv(test_path, sep='\t', header=None)

    # First column is label, rest is time series [cite: 15]
    # Standardize labels to 0 and 1
    y_train = torch.tensor(train_df.iloc[:, 0].values, dtype=torch.long)
    X_train = torch.tensor(train_df.iloc[:, 1:].values, dtype=torch.float32)

    y_test = torch.tensor(test_df.iloc[:, 0].values, dtype=torch.long)
    X_test = torch.tensor(test_df.iloc[:, 1:].values, dtype=torch.float32)

    # Note: Earthquakes labels are often 0/1 or 1/2.
    # Ensure they start at 0 for CrossEntropyLoss
    if y_train.min() > 0:
        y_train = y_train - y_train.min()
        y_test = y_test - y_test.min()

    train_ds = TensorDataset(X_train, y_train)
    test_ds = TensorDataset(X_test, y_test)

    return DataLoader(train_ds, batch_size=batch_size, shuffle=True), \
        DataLoader(test_ds, batch_size=batch_size)