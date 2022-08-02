from model import HyperPrior
import argparse
from utils import load_checkpoint
from PIL import Image
from torchvision import transforms
import json
from env import AttrDict, build_env
import os
import torch

parser_ = argparse.ArgumentParser()
'''
    '--model_name': Name of the model
    '--test_dir': Test data dir
    '--config_file': Path of your config file
    '--lambda_': The lambda setting for RD loss
    '--checkpoint_path: The path of models
'''

parser_.add_argument('--model_name', default='image_compressor', type=str)
parser_.add_argument('--config_file', default="./configs/config.json", type=str)
parser_.add_argument('--lambda_', default=0.0483, type=float)
parser_.add_argument('--checkpoint_path',
                     default="./checkpoint/image_compressor/models/lambda0.013_batchsize64_image_compressor_00280000",
                     type=str)
a = parser_.parse_args()
with open(a.config_file) as f:
    data = f.read()
json_config = json.loads(data)
h = AttrDict(json_config)
build_env(a.config_file, 'config.json', os.path.join(a.checkpoint_path, a.model_name))

compressor = HyperPrior(a, h, 0)
# 导入训练好模型参数到之前的模型实例中
state_dict_com = load_checkpoint(r"./checkpoint/lambda0.0483_batchsize4_image_compressor_00504000",
                                 torch.device('cuda:0'))
compressor.load_state_dict(state_dict_com['compressor'])
# 读入一幅图像
compressor = compressor.to(torch.device('cuda:0'))  # 模型传入GPU
# for img_file in os.listdir(r"\dataset\KoDak"):
#     img_path = os.path.join(r"\dataset\KoDak", img_file)
#     image = Image.open(img_path).convert('RGB')
#     # 转换成Tensor格式，才能进入网络
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     img = transform(image)
#     img = img.unsqueeze(0).cuda()  # 要加一个纬度，用来匹配batch的纬度
#     # 得到网络forwar的输出
#     x_hat, bpp_y, bpp_z = compressor.inference(img)

image = Image.open(r"\dataset\KoDak\kodim03.png").convert('RGB')
# 转换成Tensor格式，才能进入网络
transform = transforms.Compose([
    transforms.ToTensor()
])
img = transform(image)
img = img.unsqueeze(0).cuda()  # 要加一个纬度，用来匹配batch的纬度
# 得到网络forwar的输出
x_hat, bpp_y, bpp_z = compressor.inference(img)