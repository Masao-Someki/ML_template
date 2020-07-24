import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        self.criteria = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.criteria(x[0], x[1])

