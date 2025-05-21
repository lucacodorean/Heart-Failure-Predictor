import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size=10, num_classes=3):
        super().__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)