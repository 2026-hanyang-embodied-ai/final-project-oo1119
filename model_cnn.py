# model_cnn.py
"""1D CNN + MLP head for raw-signal fault isolation."""
import torch
import torch.nn as nn


class CnnMlpFaultIsolation(nn.Module):
    """
    Input : (batch, n_signals, n_samples)  e.g. (N, 9, 500)
    Output: (batch, num_classes) logits

    Architecture:
        Conv block 1: (9, 500) -> (64, 125)
        Conv block 2: (64, 125) -> (128, 31)
        Conv block 3: (128, 31) -> (256, 1)  via AdaptiveAvgPool
        MLP head    : 256 -> 128 -> num_classes
    """

    def __init__(self, n_signals: int = 9, num_classes: int = 5):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(n_signals, 64, kernel_size=16, padding=8),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(4),

            nn.Conv1d(64, 128, kernel_size=8, padding=4),
            nn.BatchNorm1d(128), nn.ReLU(), nn.MaxPool1d(4),

            nn.Conv1d(128, 256, kernel_size=4, padding=2),
            nn.BatchNorm1d(256), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.1),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(x))
