# from __future__ import print_function, division
# import os
# import os.path
# from PIL import Image
# # from util import is_image_file, load_img
# import pandas as pd
# from torch.utils.data import Dataset, DataLoader
# from os import listdir
# from os.path import join
# from PIL import Image
# import torch.utils.data as data
# # import torchvision.transforms as transforms
# import random
# from util import is_image_file
# from torchvision.transforms import Compose, ToTensor
# import torch
# import matplotlib.pyplot as plt
#
#
# def transform():
#     return Compose([
#         ToTensor(),
#     ])
#
#
# def make_dataset(txt_file):
#     image = []
#     path_all = pd.read_table(txt_file, header=None, delim_whitespace=True)
#
#     for i in range(len(path_all)):
#         img_lr = path_all.iloc[i, 0]
#         img_hr = path_all.iloc[i, 1]
#         img_sr = path_all.iloc[i, 2]
#
#         if is_image_file(img_lr):
#             item = (img_lr, img_hr, img_sr)
#             image.append(item)
#     return image
#
#
# def get_patch(input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output ,patch_size, scale, ix=-1, iy=-1):
#     (ih, iw) = input.size
#     (th, tw) = (scale * ih, scale * iw)
#
#     patch_mult = scale  # if len(scale) > 1 else 1
#     tp = patch_mult * patch_size
#     ip = tp // scale
#
#     if ix == -1:
#         ix = random.randrange(0, iw - ip + 1)
#     if iy == -1:
#         iy = random.randrange(0, ih - ip + 1)
#
#     (tx, ty) = (scale * ix, scale * iy)
#
#     input = input.crop((iy, ix, iy + ip, ix + ip))
#     input_n1 = input_n1.crop((iy, ix, iy + ip, ix + ip))
#     input_n2 = input_n2.crop((iy, ix, iy + ip, ix + ip)) # [:, iy:iy + ip, ix:ix + ip]
#     input_n3 = input_n3.crop((iy, ix, iy + ip, ix + ip))
#     input_p1 = input_p1.crop((iy, ix, iy + ip, ix + ip))
#     input_p2 = input_p2.crop((iy, ix, iy + ip, ix + ip))
#     input_p3 = input_p3.crop((iy, ix, iy + ip, ix + ip))
#     output = output.crop((ty, tx, ty + tp, tx + tp))
#     # img_sr = img_sr.crop((ty, tx, ty + tp, tx + tp))# [:, ty:ty + tp, tx:tx + tp]
#     plt.imshow(input_n3)
#     plt.show()
#     plt.imshow(input_n2)
#     plt.show()
#     plt.imshow(input_n1)
#     plt.show()
#     plt.imshow(input)
#     plt.show()
#     plt.imshow(input_p1)
#     plt.show()
#     plt.imshow(input_p2)
#     plt.show()
#     plt.imshow(input_p3)
#     plt.show()
#     plt.imshow(output)
#     plt.show()
#
#     info_patch = {
#         'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
#
#     # return input_n3.convert('RGB'), input_n2.convert('RGB'),input_n1.convert('RGB'),input.convert('RGB'), input_p1.convert('RGB'),input_p2.convert('RGB'),input_p3.convert('RGB'),output.convert('RGB'), info_patch
#     return input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output, info_patch
#
#
# def load_img(filepath):
#     img = Image.open(filepath)
#     # img = Image.open(filepath)
#     # img = img.convert('RGB')
#     return img
#
# # def default_loader(path):
# #     return Image.open(path).convert('RGB')
#
# class CelebADataset(Dataset):
#
#     def __init__(self, txt_file, transform=transform()):
#         #classes, class_to_idx = find_classes(root)
#
#         imgs = make_dataset(txt_file)
#         if len(imgs) == 0:
#             raise(RuntimeError("Found 0 images in subfolders of: " + "\n"
#                                "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
#
#         # self.root = root
#         self.imgs = imgs
#         # self.transform = transform
#         # self.target_transform = target_transform
#         # self.loader = load_img(filepath)
#         self.aa = 0
#
#         # transform_list = [transforms.ToTensor()]
#         # self.transform = transforms.Compose(transform_list)
#         self.transform = transform
#
#     def __getitem__(self, index):
#
#
#         # img_lr, img_hr, img_sr = self.imgs[self.aa]
#         # self.aa = self.aa + 1
#         index_p3 = index - 3
#         index_p2 = index - 2
#         index_p1 = index - 1
#         index_n1 = index + 1
#         index_n2 = index + 2
#         index_n3 = index + 3
#
#         if index % 100 == 0:
#             index_p3 = index
#             index_p2 = index
#             index_p1 = index
#
#         if index % 100 == 1:
#             index_p3 = index - 1
#             index_p2 = index - 1
#         if index % 100 == 2:
#             index_p3 = index - 2
#
#         if index % 100 == 99:
#             index_n1 = index
#             index_n2 = index
#             index_n3 = index
#
#         if index % 100 == 98:
#             index_n2 = index + 1
#             index_n3 = index + 1
#         if index % 100 == 97:
#             index_n3 = index + 2
#
#         img_lr_p3, img_hr_p3, img_sr_p3 = self.imgs[index_p3]
#         img_lr_p2, img_hr_p2, img_sr_p2 = self.imgs[index_p2]
#         img_lr_p1, img_hr_p1, img_sr_p1 = self.imgs[index_p1]
#         img_lr, img_hr, img_sr = self.imgs[index]
#         img_lr_n1, img_hr_n1, img_sr_n1 = self.imgs[index_n1]
#         img_lr_n2, img_hr_n2, img_sr_n2 = self.imgs[index_n2]
#         img_lr_n3, img_hr_n3, img_sr_n3 = self.imgs[index_n3]
#
#         input_p3 = load_img(img_lr_p3)
#         input_p2 = load_img(img_lr_p2)
#         input_p1 = load_img(img_lr_p1)
#         input = load_img(img_lr)
#         input_n1 = load_img(img_lr_n1)
#         input_n2 = load_img(img_lr_n2)
#         input_n3 = load_img(img_lr_n3)
#         output = load_img(img_hr)
#
#         # sr = load_img(img_hr)
#         input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output, _ = get_patch(input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output, patch_size=64, scale=4)
#
#         input = self.transform(input)
#         input_n1 = self.transform(input_n1)
#         input_n2 = self.transform(input_n2)
#         input_n3 = self.transform(input_n3)
#         input_p1 = self.transform(input_p1)
#         input_p2 = self.transform(input_p2)
#         input_p3 = self.transform(input_p3)
#         output = self.transform(output)
#
#         input = input.unsqueeze(0)
#         input_n1 = input_n1.unsqueeze(0)
#         input_n2 = input_n2.unsqueeze(0)
#         input_n3 = input_n3.unsqueeze(0)
#         input_p1 = input_p1.unsqueeze(0)
#         input_p2 = input_p2.unsqueeze(0)
#         input_p3 = input_p3.unsqueeze(0)
#         input = torch.cat((input_p3, input_p2, input_p1, input, input_n1, input_n2, input_n3),0)
#
#
#         # sr = self.transform(sr)
#         # print(index)
#
#
#         return input, output
#
#     def __len__(self):
#         return len(self.imgs)
from __future__ import print_function, division
import os
import os.path
from PIL import Image
# from util import is_image_file, load_img
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from os import listdir
from os.path import join
from PIL import Image
import torch.utils.data as data
# import torchvision.transforms as transforms
import random
from util import is_image_file
from torchvision.transforms import Compose, ToTensor
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np


def transform():
    return Compose([
        ToTensor(),
    ])


def make_dataset(txt_file):
    image = []
    path_all = pd.read_table(txt_file, header=None, delim_whitespace=True)

    for i in range(len(path_all)):
        img_lr = path_all.iloc[i, 0]
        img_hr = path_all.iloc[i, 1]
        # img_sr = path_all.iloc[i, 2]

        if is_image_file(img_lr):
            item = (img_lr, img_hr)
            image.append(item)
    return image

# def modcrop(img_in, scale):
#     """img_in: Numpy, HWC or HW"""
#     img = np.copy(img_in)
#     if img.ndim == 2:
#         H, W = img.shape
#         H_r, W_r = H % scale, W % scale
#         img = img[:H - H_r, :W - W_r]
#     elif img.ndim == 3:
#         H, W, C = img.shape
#         H_r, W_r = H % scale, W % scale
#         img = img[:H - H_r, :W - W_r, :]
#     else:
#         raise ValueError('Wrong img ndim: [{:d}].'.format(img.ndim))
#     return img
# def get_patch(input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output ,patch_size, scale, ix=-1, iy=-1):
#     (ih, iw) = input.size
#     (th, tw) = (scale * ih, scale * iw)
#
#     patch_mult = scale  # if len(scale) > 1 else 1
#     tp = patch_mult * patch_size
#     ip = tp // scale
#
#     if ix == -1:
#         ix = random.randrange(0, iw - ip + 1)
#     if iy == -1:
#         iy = random.randrange(0, ih - ip + 1)
#
#     (tx, ty) = (scale * ix, scale * iy)
#
#     input = input.crop((iy, ix, iy + ip, ix + ip))
#     input_n1 = input_n1.crop((iy, ix, iy + ip, ix + ip))
#     input_n2 = input_n2.crop((iy, ix, iy + ip, ix + ip)) # [:, iy:iy + ip, ix:ix + ip]
#     input_n3 = input_n3.crop((iy, ix, iy + ip, ix + ip))
#     input_p1 = input_p1.crop((iy, ix, iy + ip, ix + ip))
#     input_p2 = input_p2.crop((iy, ix, iy + ip, ix + ip))
#     input_p3 = input_p3.crop((iy, ix, iy + ip, ix + ip))
#     output = output.crop((ty, tx, ty + tp, tx + tp))
#     # img_sr = img_sr.crop((ty, tx, ty + tp, tx + tp))# [:, ty:ty + tp, tx:tx + tp]
#
#
#     info_patch = {
#         'ix': ix, 'iy': iy, 'ip': ip, 'tx': tx, 'ty': ty, 'tp': tp}
#
#     # return input_n3.convert('RGB'), input_n2.convert('RGB'),input_n1.convert('RGB'),input.convert('RGB'), input_p1.convert('RGB'),input_p2.convert('RGB'),input_p3.convert('RGB'),output.convert('RGB'), info_patch
#     return input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output, info_patch


# def load_img(filepath):
#     img = Image.open(filepath)
#     # img = Image.open(filepath)
#     # img = img.convert('RGB')
#     return img


def read_img(path):
    """read image by cv2 or from lmdb
    return: Numpy float32, HWC, BGR, [0,1]"""
    # img
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)

    img = img.astype(np.float32) / 255.
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)
    # some images have 4 channels
    if img.shape[2] > 3:
        img = img[:, :, :3]
    return img

# def default_loader(path):
#     return Image.open(path).convert('RGB')

class CelebADataset(Dataset):

    def __init__(self, txt_file, transform=transform()):
        #classes, class_to_idx = find_classes(root)

        imgs = make_dataset(txt_file)
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))

        # self.root = root
        self.imgs = imgs
        # self.transform = transform
        # self.target_transform = target_transform
        # self.loader = load_img(filepath)
        self.aa = 0

        # transform_list = [transforms.ToTensor()]
        # self.transform = transforms.Compose(transform_list)
        self.transform = transform

    def __getitem__(self, index):



        # img_lr, img_hr, img_sr = self.imgs[self.aa]
        # self.aa = self.aa + 1
        index_p3 = index - 3
        index_p2 = index - 2
        index_p1 = index - 1
        index_n1 = index + 1
        index_n2 = index + 2
        index_n3 = index + 3

        if index % 100 == 0:
            index_p3 = index
            index_p2 = index
            index_p1 = index

        if index % 100 == 1:
            index_p3 = index - 1
            index_p2 = index - 1
        if index % 100 == 2:
            index_p3 = index - 2

        if index % 100 == 99:
            index_n1 = index
            index_n2 = index
            index_n3 = index

        if index % 100 == 98:
            index_n2 = index + 1
            index_n3 = index + 1
        if index % 100 == 97:
            index_n3 = index + 2


        # img_lr_p3, img_hr_p3 = self.imgs[index_p3]
        img_lr_p2, img_hr_p2 = self.imgs[index_p2]
        img_lr_p1, img_hr_p1 = self.imgs[index_p1]
        img_lr, img_hr = self.imgs[index]
        img_lr_n1, img_hr_n1 = self.imgs[index_n1]
        img_lr_n2, img_hr_n2 = self.imgs[index_n2]
        # img_lr_n3, img_hr_n3 = self.imgs[index_n3]




        # input_p3 = read_img(img_lr_p3)
        input_p2 = read_img(img_lr_p2)
        input_p1 = read_img(img_lr_p1)
        input = read_img(img_lr)
        input_n1 = read_img(img_lr_n1)
        input_n2 = read_img(img_lr_n2)
        # input_n3 = read_img(img_lr_n3)
        output = read_img(img_hr)

        H, W, C = input.shape
        LQ_size = 128

        # randomly crop
        rnd_h = random.randint(0, max(0, H - LQ_size))
        rnd_w = random.randint(0, max(0, W - LQ_size))
        input = input[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        # input_p3 = input_p3[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        input_p2 = input_p2[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        input_p1 = input_p1[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        input_n1 = input_n1[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        input_n2 = input_n2[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        # input_n3 = input_n3[rnd_h:rnd_h + LQ_size, rnd_w:rnd_w + LQ_size, :]
        rnd_h_GT, rnd_w_GT = int(rnd_h * 4), int(rnd_w * 4)
        output = output[rnd_h_GT:rnd_h_GT + 512, rnd_w_GT:rnd_w_GT + 512, :]

        img_LQ_l = []
        # img_LQ_l.append(input_p3)
        img_LQ_l.append(input_p2)
        img_LQ_l.append(input_p1)
        img_LQ_l.append(input)
        img_LQ_l.append(input_n1)
        img_LQ_l.append(input_n2)
        # img_LQ_l.append(input_n3)
        img_LQs = np.stack(img_LQ_l, axis=0)

        output = output[:, :, [2, 1, 0]]
        input = img_LQs[:, :, :, [2, 1, 0]]
        output = torch.from_numpy(np.ascontiguousarray(np.transpose(output, (2, 0, 1)))).float()
        input = torch.from_numpy(np.ascontiguousarray(np.transpose(input,
                                                                     (0, 3, 1, 2)))).float()


        # sr = load_img(img_hr)
        # input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output, _ = get_patch(input_n3, input_n2, input_n1, input, input_p1, input_p2, input_p3, output, patch_size=64, scale=4)

        # input = self.transform(input)
        # input_n1 = self.transform(input_n1)
        # input_n2 = self.transform(input_n2)
        # input_n3 = self.transform(input_n3)
        # input_p1 = self.transform(input_p1)
        # input_p2 = self.transform(input_p2)
        # input_p3 = self.transform(input_p3)
        # output = self.transform(output)
        #
        # input = input.unsqueeze(0)
        # input_n1 = input_n1.unsqueeze(0)
        # input_n2 = input_n2.unsqueeze(0)
        # input_n3 = input_n3.unsqueeze(0)
        # input_p1 = input_p1.unsqueeze(0)
        # input_p2 = input_p2.unsqueeze(0)
        # input_p3 = input_p3.unsqueeze(0)
        # input = torch.cat((input_p3, input_p2, input_p1, input, input_n1, input_n2, input_n3),0)



        # sr = self.transform(sr)
        # print(index)


        return input, output

    def __len__(self):
        return len(self.imgs)
