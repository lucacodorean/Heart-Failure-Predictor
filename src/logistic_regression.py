import torch
import torch.nn as nn

class LogisticRegression(nn.Module):
    def __init__(self, input_size=10, num_classes=3):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(input_size, num_classes))
        self.b = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x):
        return x @ self.w + self.b