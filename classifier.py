import torch
from torch import nn

class LinearClassifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearClassifier, self).__init__()
        self.fcl = nn.Linear(input_dim, output_dim)

    def forward(self, input):
        return self.fcl(input)

