import torch
import argparse
from utils import load_checkpoint
from PIL import Image
from torchvision import transforms
import json
from env import AttrDict, build_env
import os
from model import HyperPrior

# from skimage.metrics import peak_signal_noise_ratio
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--model_name', default='image_compressor', type=str)
parser.add_argument('--training_dir', default=r'./dataset/vimeo/train', type=str)
parser.add_argument('--validation_dir', default=r'./dataset/vimeo/test', type=str)
parser.add_argument('--checkpoint_path', default='./checkpoint', type=str)
parser.add_argument('--config_file', default=r'./configs/config.json', type=str)
parser.add_argument('--training_epochs', default=3000, type=int)
parser.add_argument('--stdout_interval', default=5, type=int)
parser.add_argument('--checkpoint_interval', default=5000, type=int)
parser.add_argument('--summary_interval', default=100, type=int)
parser.add_argument('--validation_interval', default=1000, type=int)
parser.add_argument('--fine_tuning', default=False, type=bool)
parser.add_argument('--Lambda', default=0.0067, type=float)

a = parser.parse_args()

with open(a.config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
build_env(a.config_file, 'config.json', os.path.join(a.checkpoint_path, a.model_name))

device = torch.device('cuda:0')
# 以上的都不用去理解
# ————————————————————————————————————————————————
# 初始化一个类实例，这个实例里包含网络结构
compressor = HyperPrior(a, h, 0)
# 导入训练好模型参数到之前的模型实例中
state_dict_com = load_checkpoint(r"./checkpoint/HyperPrior/HyperPrior_000000", device)
compressor.load_state_dict(state_dict_com['compressor'])
# 读入一幅图像
image = Image.open(r"Anyimage.png").convert('RGB')
# 转换成Tensor格式，才能进入网络
transform = transforms.Compose([
    transforms.ToTensor()
])
img = transform(image)
img = img.unsqueeze(0).cuda() # 要加一个纬度，用来匹配batch的纬度
compressor = compressor.to(device) # 模型传入GPU
# 得到网络forwar的输出
loss, bpp_y, bpp_z, disstortion, x_hat = compressor(img)
# 打印相关信息
print("loss:{}  bpp_y:{}  bpp_z:{}  disstortion:{}".format(loss,bpp_y,bpp_z,disstortion))
# 重构图像x_hat转换成PIL格式图像，保存
inv_transform = transforms.ToPILImage()
reco_img = inv_transform(x_hat)
reco_img.save("reco_img.png")