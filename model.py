import torch
from torch import nn
import torch.nn.functional as F
from modules import *
from time import time


class HyperPrior(nn.Module):
    def __init__(self, a, h, rank, N = 128, M = 192):
        super(HyperPrior, self).__init__()
        self.a = a
        self.h = h
        self.N = N
        self.M = M
        self.device = torch.device('cuda:{}'.format(rank))
        # Encoders and decoders
        self.g_a = Analysis(N=self.N, M=self.M)
        self.g_s = Synthesis(N=self.N, M=self.M)
        self.h_a = AnalysisPrior(N=self.N, M=self.M)
        self.h_s = SynthesisPrior(N=self.N, M=self.M)
        # Entropy model
        self.factorized_model = FactorizedModel(num_channel=self.N, K=4)
        self.gaussian_model = GaussianModel()
        # Entropy codec
        self.entropy_coder_factorized = EntropyCoder(self.factorized_model)
        self.entropy_coder_gaussian = EntropyCoderGaussian(self.gaussian_model)

    def quantize(self, input_, is_tain=True):
        if is_tain:
            uniform_noise = torch.nn.init.uniform_(torch.zeros_like(input_), -0.5, 0.5)
            if torch.cuda.is_available():
                uniform_noise = uniform_noise.to(self.device)
            return input_ + uniform_noise
        else:
            return torch.round(input_)

    def forward(self, input_):
        x = input_

        y = self.g_a(x)
        y_hat = self.quantize(y, is_tain=True)
        x_hat = torch.clamp(self.g_s(y_hat), min=0, max=1)

        z = self.h_a(torch.abs(y))
        z_hat = self.quantize(z, is_tain=True)
        scale = self.h_s(z_hat)

        bits_z = torch.sum(torch.clamp(-torch.log2(self.factorized_model.likelihood(z_hat)), min=0, max=50))
        bpp_z = bits_z / (input_.shape[0] * input_.shape[2] * input_.shape[3])
        bits_y = torch.sum(torch.clamp(-torch.log2(self.gaussian_model.likelihood(y_hat, scale)), min=0, max=50))
        bpp_y = bits_y / (input_.shape[0] * input_.shape[2] * input_.shape[3])

        disstortion = torch.mean((x - x_hat) ** 2)

        loss = (bpp_y + bpp_z) + self.a.lambda_ * (255 ** 2) * disstortion
        return loss, bpp_y, bpp_z, disstortion, x_hat

    def inference(self, input_):

        time_enc_start = time()
        x = input_
        y = self.g_a(x)
        y_hat = self.quantize(y, is_tain=False)

        z = self.h_a(torch.abs(y))
        z_hat = self.quantize(z, is_tain=False)
        torch.use_deterministic_algorithms(mode=True)
        scale = self.h_s(z_hat)
        torch.use_deterministic_algorithms(mode=False)

        stream_z, side_info_z = self.entropy_coder_factorized.compress(z_hat)
        stream_y, side_info_y = self.entropy_coder_gaussian.compress(y_hat, scale)
        time_enc_end = time()

        time_dec_start = time()
        z_hat_dec = self.entropy_coder_factorized.decompress(stream_z, side_info_z, self.device)
        assert torch.equal(z_hat, z_hat_dec), "Entropy code decode for z_hat not consistent !"
        torch.use_deterministic_algorithms(mode=True)
        scale_dec = self.h_s(z_hat_dec)
        torch.use_deterministic_algorithms(mode=False)
        assert torch.equal(scale, scale_dec), "Scale codec not consistent!"

        y_hat_dec = self.entropy_coder_gaussian.decompress(stream_y, side_info_y, scale_dec, self.device)
        assert torch.equal(y_hat, y_hat_dec), "y_hat not equal to y_hat_dec !"
        x_hat = torch.clamp(self.g_s(y_hat_dec), min=0, max=1)
        time_dec_end = time()
        # print("{:.4f}, {:.4f}".format((time_enc_end - time_enc_start), (time_dec_end - time_dec_start)))

        _ = 0
        bpp_y = len(stream_y) * 8 / (input_.shape[0] * input_.shape[2] * input_.shape[3])
        bpp_z = len(stream_z) * 8 / (input_.shape[0] * input_.shape[2] * input_.shape[3])
        return x_hat, bpp_y, bpp_z, (time_enc_end - time_enc_start), (time_dec_end - time_dec_start)


if __name__ == '__main__':
    model = HyperPrior(N=128, M=192)

