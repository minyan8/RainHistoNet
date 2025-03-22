## Learning Enriched Features for Fast Image Restoration and Enhancement
## Syed Waqas Zamir, Aditya Arora, Salman Khan, Munawar Hayat, Fahad Shahbaz Khan, Ming-Hsuan Yang, and Ling Shao
## IEEE Transactions on Pattern Analysis and Machine Intelligence (TPAMI)
## https://www.waqaszamir.com/publication/zamir-2022-mirnetv2/

import numpy as np
import os
import argparse
from tqdm import tqdm

import torch.nn as nn
import torch
import torch.nn.functional as F
import utils

from natsort import natsorted
from glob import glob
from basicsr.models.archs.UHDM_arch import UHDM
from skimage import img_as_ubyte

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

parser = argparse.ArgumentParser(description='Image Enhancement using MIRNet-v2')

parser.add_argument('--input_dir', default='/home/min/Documents/ntire25/raindrop/data/RainDrop', type=str, help='Directory of validation images')
parser.add_argument('--result_dir', default='../results', type=str, help='Directory for results')
parser.add_argument('--weights', default='/home/min/Documents/ntire25/raindrop/SYSU-FVL-T2/experiments/sysu_hist_rain/models/net_g_5000.pth', type=str, help='Path to weights')
parser.add_argument('--dataset', default='Raindrop', type=str, help='Test Dataset')

args = parser.parse_args()


####### Load yaml #######
yaml_file = 'Options/rainHistoNet.yml'
weights = args.weights

import yaml

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

x = yaml.load(open(yaml_file, mode='r'), Loader=Loader)

s = x['network_g'].pop('type')
##########################

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            # if type(x) == tuple:
            #     x = torch.stack(x)
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)

model_restoration = UHDM(**x['network_g'])

checkpoint = torch.load(weights)
model_restoration.load_state_dict(checkpoint['params'])
print("===>Testing using weights: ",weights)
model_restoration.cuda()
model_restoration = nn.DataParallel(model_restoration)
model_restoration.eval()

# # code for profiling
# from thop import profile
# dummy_input = torch.randn(1, 3, 720, 480).cuda()
# model_for_profile = model_restoration.module if isinstance(model_restoration, torch.nn.DataParallel) else model_restoration
# flops, params = profile(model_for_profile, inputs=(dummy_input,))
# # Convert to GFLOPs
# print(f"GFLOPs: {flops / 1e9:.2f}, Parameters: {params / 1e6:.2f}M")

factor = 16
dataset = args.dataset
result_dir  = os.path.join(args.result_dir, dataset)
os.makedirs(result_dir, exist_ok=True)

# input_dir = os.path.join(args.input_dir, 'LL')
# input_paths = natsorted(glob(os.path.join(input_dir, '*.png')) + glob(os.path.join(input_dir, '*.jpg')))

# target_dir = os.path.join(args.input_dir, 'GT')
# target_paths = natsorted(glob(os.path.join(target_dir, '*.png')) + glob(os.path.join(target_dir, '*.jpg')))

# for unpaired data
input_paths = natsorted(glob(os.path.join(args.input_dir, '*.png')) + glob(os.path.join(args.input_dir, '*.JPG')))
target_paths = natsorted(glob(os.path.join(args.input_dir, '*.png')) + glob(os.path.join(args.input_dir, '*.JPG')))


psnr = []
ssim = []
with torch.inference_mode():
    for inp_path, tar_path in tqdm(zip(input_paths,target_paths), total=len(target_paths)):
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        img = np.float32(utils.load_img(inp_path))/255.
        target = np.float32(utils.load_img(tar_path))/255.

        img = torch.from_numpy(img).permute(2,0,1)
        input_ = img.unsqueeze(0).cuda()

        # Padding in case images are not multiples of 4
        h,w = input_.shape[2], input_.shape[3]
        print("h,w", h, w)
        H,W = ((h+factor)//factor)*factor, ((w+factor)//factor)*factor
        padh = H-h if h%factor!=0 else 0
        padw = W-w if w%factor!=0 else 0
        input_ = F.pad(input_, (0,padw,0,padh), 'reflect')
        print(inp_path)


        if h < 3000 and w < 3000:
            restored = model_restoration(input_)
            # Unpad images to original dimensions
            restored = restored[0][:,:,:h,:w]
        else:
            # split and test
            input_1 = input_[:, :, :, 0::2]
            input_2 = input_[:, :, :, 1::2]

            input_1_1 = input_1[:, :, 0::2, :]
            input_1_2 = input_1[:, :, 1::2, :]

            input_2_1 = input_2[:, :, 0::2, :]
            input_2_2 = input_2[:, :, 1::2, :]

            print(input_1_1.shape)
            restored_1_1 = model_restoration(input_1_1)[0]
            print(input_1_2.shape)
            restored_1_2 = model_restoration(input_1_2)[0]

            print(input_2_1.shape)
            restored_2_1 = model_restoration(input_2_1)[0]
            print(input_2_2.shape)
            restored_2_2 = model_restoration(input_2_2)[0]

            restored = torch.zeros_like(input_)

            restored[:, :, 0::2, 0::2] = restored_1_1
            restored[:, :, 1::2, 0::2] = restored_1_2

            restored[:, :, 0::2, 1::2] = restored_2_1
            restored[:, :, 1::2, 1::2] = restored_2_2


        

        restored = torch.clamp(restored,0,1).cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()

        psnr.append(utils.PSNR(target, restored))
        ssim.append(utils.calculate_ssim(img_as_ubyte(target), img_as_ubyte(restored)))

        utils.save_img((os.path.join(result_dir, os.path.splitext(os.path.split(inp_path)[-1])[0]+'.png')), img_as_ubyte(restored))

psnr = np.mean(np.array(psnr))
ssim = np.mean(np.array(ssim))
print("PSNR: %.4f " %(psnr))
print("SSIM: %.4f " %(ssim))

