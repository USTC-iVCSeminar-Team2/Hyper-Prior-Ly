import torch
from model import HyperPrior

class Args():
    def __init__(self):
        self.lambda_ = 0.0067

if __name__ == '__main__':
    a = Args()
    model = HyperPrior(a=a, h='', rank='', N=128, M=192)
    input_ = torch.randn(size=(1, 3, 768, 512))
    loss, bpp_y, bpp_z, disstortion, x_hat = model(input_)
    model.inference(input_)
    loss.backward()
    _ = 0
