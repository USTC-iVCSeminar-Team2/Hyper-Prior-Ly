import torch
from torch import nn
import torch.nn.functional as F
from modules import SetMinBoundary


class GaussianModel(nn.Module):
    def __init__(self):
        super(GaussianModel, self).__init__()

    def standardized_cumulative(self, input_):
        return 1.0 - 0.5 * torch.erfc((2 ** -0.5) * input_)     # c(x) = 1 - c(-x) = 0.5 * erfc(-(2**-0.5) * x)

    def forward(self, input_, scale):
        assert input_.shape[0:3] == scale.shape[0:3], "Shape dismatch between y and gaussian scale"
        scale = SetMinBoundary.apply(scale, 1e-6)
        input_ = input_ / scale
        cumul = self.standardized_cumulative(input_)
        return cumul

    def likelihood(self, input_, scale):
        likelihood_ = self.forward(input_ + 0.5, scale) - self.forward(input_ - 0.5, scale) + 1e-6
        return likelihood_


if __name__ == '__main__':
    gaussian_model = GaussianModel()

    x = torch.nn.init.normal_(torch.Tensor(4 ,128, 16, 16), mean=0, std=2)
    scale = torch.ones(x.shape, dtype=torch.float32) * 2

    output = gaussian_model(x, scale)
    output_ = gaussian_model.likelihood(x, scale)
    _ = 0

