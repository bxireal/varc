from __future__ import print_function
import argparse
import os
from math import log10
import time, math
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np
import torch
from torch.autograd import Variable
import torchvision.transforms as transforms
from PIL import Image
from util import is_image_file, load_img, save_img
from networks import _NetBFSR
import os.path as osp
import glob
import logging
# import sys
import torch.nn.functional as F
import torch
from os.path import join
import cv2
import RRDBNet_arch as arch
import archs.EDVR_arch as EDVR_arch
# import importlib
# importlib.reload(sys)
# sys.setdefaultencoding('utf-8')

# Testing settings
parser = argparse.ArgumentParser(description='pix2pix-PyTorch-implementation')
parser.add_argument('--dataset', default="test", help='facades')
parser.add_argument("--model", default="checkpoint/data/", type=str, help="model path")
parser.add_argument('--cuda', default="true", action='store_true', help='use cuda')
torch.cuda.set_device(2)

opt = parser.parse_args()
print(opt)


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]
    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)



opt = parser.parse_args()
cuda = opt.cuda
if cuda and not torch.cuda.is_available():
    raise Exception("No GPU found, please run without --cuda")


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)

model = torch.load(opt.model + "netG_model_epoch_" + str(idx_mod + 1) + ".pth")["netG"]

model.eval()
model = model.cuda(2)
psnr_bicu = 0
psnr_bfsr = 0
ssim_nn = 0
aa = 0
count = 0
crit = []
elapsed_time = 0


test_dataset_folder = '/data/JXR/test_1'
# image_filenames = [x for x in os.listdir(image_hdir) if is_image_file(x)]
save_folder = ''


transform_list = [transforms.ToTensor()]
transform = transforms.Compose(transform_list)

subfolder_name_l = []

subfolder_l = sorted(glob.glob(osp.join(test_dataset_folder, '*')))

for subfolder in subfolder_l:
        subfolder_name = osp.basename(subfolder)
        subfolder_name_l.append(subfolder_name)
        save_subfolder = osp.join(save_folder, subfolder_name)

        img_path_l = sorted(glob.glob(osp.join(subfolder, '*')))
        max_idx = len(img_path_l)

        mkdirs(save_subfolder)

        #### read LQ and GT images

        for i in range(1,101):
            index_p3 = i - 3
            index_p2 = i - 2
            index_p1 = i - 1
            index_n1 = i + 1
            index_n2 = i + 2
            index_n3 = i + 3
            if i == 1:
                index_p3 = i
                index_p2 = i
                index_p1 = i

            if i == 2:
                index_p3 = i - 1
                index_p2 = i - 1

            if i == 3:
                index_p3 = i - 2

            if i == 100:
                index_n1 = i
                index_n2 = i
                index_n3 = i

            if i == 99:
                index_n2 = i + 1
                index_n3 = i + 1

            if i == 98:
                index_n3 = i + 2

            img_name =  subfolder_name +  '_' + str(i) + '.png'
            img_namep3 = subfolder_name + '_' + str(index_p3) + '.png'
            img_namep2 = subfolder_name + '_' + str(index_p2) + '.png'
            img_namep1 = subfolder_name + '_' + str(index_p1) + '.png'
            img_namen1 = subfolder_name + '_' + str(index_n1) + '.png'
            img_namen2 = subfolder_name + '_' + str(index_n2) + '.png'
            img_namen3 = subfolder_name + '_' + str(index_n3) + '.png'

           

            img_input = transform(load_img(join(subfolder,img_name)))
            img_inputp3 = transform(load_img(join(subfolder,img_namep3)))
            img_inputp2 = transform(load_img(join(subfolder,img_namep2)))
            img_inputp1 = transform(load_img(join(subfolder,img_namep1)))
            img_inputn1 = transform(load_img(join(subfolder,img_namen1)))
            img_inputn2 = transform(load_img(join(subfolder,img_namen2)))
            img_inputn3 = transform(load_img(join(subfolder,img_namen3)))

            img_input = img_input.unsqueeze(0)
            img_inputp3 = img_inputp3.unsqueeze(0)
            img_inputp2 = img_inputp2.unsqueeze(0)
            img_inputp1 = img_inputp1.unsqueeze(0)
            img_inputn1 = img_inputn1.unsqueeze(0)
            img_inputn2 = img_inputn2.unsqueeze(0)
            img_inputn3 = img_inputn3.unsqueeze(0)

            input = torch.cat((img_inputp3,img_inputp2, img_inputp1, img_input, img_inputn1, img_inputn2, img_inputn3),0)
            input = input.unsqueeze(0).cuda(2)


            
            with torch.no_grad():
                sr = model(input)
                # sr = model(img_input)
                # sr = sr.squeeze(0)
                sr = sr.data.squeeze().float().cpu().clamp_(0,1).numpy()
                sr = np.transpose(sr[[2,1,0],:,:], (1,2,0))
                sr = (sr * 255.0).round()
                cv2.imwrite(save_subfolder + '/' + img_name, sr)
                    # image_pil.save(save_subfolder.format(img_name))
                print("Image saved as {}".format(save_subfolder.format(img_name)))
               
