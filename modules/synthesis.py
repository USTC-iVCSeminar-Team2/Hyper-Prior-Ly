import torch
from torch import nn
from .gdn import GDN


class Synthesis(nn.Module):
    def __init__(self, N = 128, M = 192):
        super(Synthesis, self).__init__()
        self.N = N
        self.M = M

        self.layer = nn.Sequential(
            nn.ConvTranspose2d(in_channels=self.M, out_channels=self.N, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            GDN(num_channel=N, inverse=True),
            nn.ConvTranspose2d(in_channels=self.N, out_channels=self.N, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            GDN(num_channel=N, inverse=True),
            nn.ConvTranspose2d(in_channels=self.N, out_channels=self.N, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
            GDN(num_channel=N, inverse=True),
            nn.ConvTranspose2d(in_channels=self.N, out_channels=3, kernel_size=(5, 5), stride=(2, 2),
                               padding=(2, 2), output_padding=(1, 1)),
        )

    def forward(self, input_):
        return self.layer(input_)
