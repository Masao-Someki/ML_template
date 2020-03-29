
import torch

class Loss(torch.nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.criterion = torch.nn.CrossEntropyLoss()

    def forward(self, x, y):
        y = y.squeeze(1)
        return self.criterion(x, y), self.criterion(x, y)

