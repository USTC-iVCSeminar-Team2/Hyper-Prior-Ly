import torch
from torch import nn
from .gdn import GDN


class Analysis(nn.Module):
    def __init__(self, N = 128, M = 192):
        super(Analysis, self).__init__()
        self.N = N
        self.M = M

        self.layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=N, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            GDN(num_channel=N, inverse=False),
            nn.Conv2d(in_channels=N, out_channels=N, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            GDN(num_channel=N, inverse=False),
            nn.Conv2d(in_channels=N, out_channels=N, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2)),
            GDN(num_channel=N, inverse=False),
            nn.Conv2d(in_channels=N, out_channels=M, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        )

    def forward(self, input_):
        return self.layer(input_)
