
import torch
import torch.nn as nn

class classifierMLP(nn.Module):
    def __init__(self):
        super(classifierMLP, self).__init__()
        self.seq_layers = nn.Sequential(
            nn.Linear(150, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 20)
        )

    def forward(self, x):
        x = self.seq_layers(x)
        return x