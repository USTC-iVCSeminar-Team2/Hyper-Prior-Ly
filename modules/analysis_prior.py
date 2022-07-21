import torch
from torch import nn


class AnalysisPrior(nn.Module):
    def __init__(self, N = 128, M = 192):
        super(AnalysisPrior, self).__init__()
        self.N = N
        self.M = M

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=self.M, out_channels=self.N, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.N, out_channels=self.N, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.N, out_channels=self.N, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        )

    def forward(self, input_):
        return self.layer(input_)
