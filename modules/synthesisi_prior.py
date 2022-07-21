import torch
from torch import nn


class SynthesisPrior(nn.Module):
    def __init__(self, N = 128, M = 192):
        super(SynthesisPrior, self).__init__()
        self.N = N
        self.M = M

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.N, out_channels=self.N, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.N, out_channels=self.N, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=self.N, out_channels=self.M, kernel_size=(3, 3), stride=(1, 1),
                               padding=(1, 1), output_padding=(0, 0)),
            nn.ReLU()
        )

    def forward(self, input_):
        return self.layer(input_)
