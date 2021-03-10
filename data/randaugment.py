# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import random

import numpy as np
import PIL
import PIL.ImageOps
import PIL.ImageEnhance
import PIL.ImageDraw
from PIL import Image
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

PARAMETER_MAX = 10

def AutoContrast(img, **kwarg):
    return PIL.ImageOps.autocontrast(img), None


def Brightness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Brightness(img).enhance(v), v


def Color(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Color(img).enhance(v), v


def Contrast(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Contrast(img).enhance(v), v


def Cutout(img, v, max_v, bias=0):
    if v == 0:
        return img
    v = _float_parameter(v, max_v) + bias
    v = int(v * min(img.size))
    return CutoutAbs(img, v)


def CutoutAbs(img, v, **kwarg):
    w, h = img.size
    x0 = np.random.uniform(0, w)
    y0 = np.random.uniform(0, h)
    x0 = int(max(0, x0 - v / 2.))
    y0 = int(max(0, y0 - v / 2.))
    x1 = int(min(w, x0 + v))
    y1 = int(min(h, y0 + v))
    xy = (x0, y0, x1, y1)
    # gray
    color = (127, 127, 127)
    img = img.copy()
    PIL.ImageDraw.Draw(img).rectangle(xy, color)
    return img, xy


def Equalize(img, **kwarg):
    return PIL.ImageOps.equalize(img), None


def Identity(img, **kwarg):
    return img, None


def Invert(img, **kwarg):
    return PIL.ImageOps.invert(img), None


def Posterize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.posterize(img, v), v


# def Rotate(img, v, max_v, bias=0):
#     v = _int_parameter(v, max_v) + bias
#     if random.random() < 0.5:
#         v = -v
#     #return img.rotate(v), v
#     img_t = transforms.ToTensor()(img)
#     H = img_t.shape[1]
#     W = img_t.shape[2]
#     theta = np.array([[np.cos(v/180*np.pi), -np.sin(v/180*np.pi), 0], [np.sin(v/180*np.pi), np.cos(v/180*np.pi), 0]]).astype(np.float)
#     theta[0,1] = theta[0,1]*H/W
#     theta[1,0] = theta[1,0]*W/H
#     #theta = np.array([[np.cos(v/180*np.pi), -np.sin(v/180*np.pi)], [np.sin(v/180*np.pi), np.cos(v/180*np.pi)]]).astype(np.float)
#     theta = torch.Tensor(theta).unsqueeze(0)

#     # meshgrid_x, meshgrid_y = torch.meshgrid(torch.arange(W, dtype=torch.float), torch.arange(H, dtype=torch.float))
#     # meshgrid = torch.stack((meshgrid_x.t()*2/W - 1, meshgrid_y.t()*2/H - 1), dim=-1).unsqueeze(0)
#     # grid = torch.matmul(meshgrid, theta)

#     # s_h = int(abs(H - W) // 2)
#     # dim_last = s_h if H > W else 0
#     # img_t = F.pad(img_t.unsqueeze(0), (dim_last, dim_last, s_h - dim_last, s_h - dim_last)).squeeze(0)
#     grid = F.affine_grid(theta, img_t.unsqueeze(0).size())
#     img_t = F.grid_sample(img_t.unsqueeze(0), grid, mode='bilinear').squeeze(0)
#     # img_t = img_t[:,:,s_h:-s_h] if H > W else img_t[:,s_h:-s_h,:]
#     img_t = transforms.ToPILImage()(img_t)
#     return img_t, v

def Rotate(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.rotate(v, resample=Image.BILINEAR, fillcolor=(127,127,127)), v

def Sharpness(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    return PIL.ImageEnhance.Sharpness(img).enhance(v), v


def ShearX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def ShearY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def Solarize(img, v, max_v, bias=0):
    v = _int_parameter(v, max_v) + bias
    return PIL.ImageOps.solarize(img, 256 - v), 256 - v


def SolarizeAdd(img, v, max_v, bias=0, threshold=128):
    v = _int_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    img_np = np.array(img).astype(np.int)
    img_np = img_np + v
    img_np = np.clip(img_np, 0, 255)
    img_np = img_np.astype(np.uint8)
    img = Image.fromarray(img_np)
    return PIL.ImageOps.solarize(img, threshold), threshold


def TranslateX(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[0])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def TranslateY(img, v, max_v, bias=0):
    v = _float_parameter(v, max_v) + bias
    if random.random() < 0.5:
        v = -v
    v = int(v * img.size[1])
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v), resample=Image.BILINEAR, fillcolor=(127,127,127)), v


def _float_parameter(v, max_v):
    return float(v) * max_v / PARAMETER_MAX


def _int_parameter(v, max_v):
    return int(v * max_v / PARAMETER_MAX)


def fixmatch_augment_pool():
    # FixMatch paper
    augs = [(AutoContrast, None, None),
            (Brightness, 0.9, 0.05),
            (Color, 0.9, 0.05),
            (Contrast, 0.9, 0.05),
            (Equalize, None, None),
            (Identity, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 0.9, 0.05),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (TranslateX, 0.3, 0),
            (TranslateY, 0.3, 0)]
    return augs


def my_augment_pool():
    # Test
    augs = [(AutoContrast, None, None),
            (Brightness, 1.8, 0.1),
            (Color, 1.8, 0.1),
            (Contrast, 1.8, 0.1),
            (Cutout, 0.2, 0),
            (Equalize, None, None),
            (Invert, None, None),
            (Posterize, 4, 4),
            (Rotate, 30, 0),
            (Sharpness, 1.8, 0.1),
            (ShearX, 0.3, 0),
            (ShearY, 0.3, 0),
            (Solarize, 256, 0),
            (SolarizeAdd, 110, 0),
            (TranslateX, 0.45, 0),
            (TranslateY, 0.45, 0)]
    return augs


class RandAugmentPC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = my_augment_pool()

    def __call__(self, img):
        ops = random.choices(self.augment_pool, k=self.n)
        for op, max_v, bias in ops:
            prob = np.random.uniform(0.2, 0.8)
            if random.random() + prob >= 1:
                img = op(img, v=self.m, max_v=max_v, bias=bias)
        img = CutoutAbs(img, 16)
        return img


class RandAugmentMC(object):
    def __init__(self, n, m):
        assert n >= 1
        assert 1 <= m <= 10
        self.n = n
        self.m = m
        self.augment_pool = fixmatch_augment_pool()

    def __call__(self, img, type='crc'):
        aug_type = {'Hflip':False, 'ShearX':1e4, 'ShearY':1e4, 'TranslateX':1e4, 'TranslateY':1e4, 'Rotate':1e4, 'CutoutAbs':1e4}
        if random.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            #aug_type.append(['Hflip', True])
            aug_type['Hflip'] = True
        if type == 'cr' or type == 'crc':
            ops = random.choices(self.augment_pool, k=self.n)
            for op, max_v, bias in ops:
                v = np.random.randint(1, self.m)
                if random.random() < 0.5:
                    img, params = op(img, v=v, max_v=max_v, bias=bias)
                    if op.__name__ in ['ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']:
                        #aug_type.append([op.__name__, params])
                        aug_type[op.__name__] = params
        if type == 'cc' or type == 'crc':
            img, params = CutoutAbs(img, min(img.size[0], img.size[1]) // 3)
            #aug_type.append([CutoutAbs.__name__, params])
            aug_type['CutoutAbs'] = params
        return img, aug_type

def affine_sample(tensor, v, type):
    # tensor: B*C*H*W
    # v: scalar, translation param
    if type == 'Rotate':
        theta = np.array([[np.cos(v/180*np.pi), -np.sin(v/180*np.pi), 0], [np.sin(v/180*np.pi), np.cos(v/180*np.pi), 0]]).astype(np.float)
    elif type == 'ShearX':
        theta = np.array([[1, v, 0], [0, 1, 0]]).astype(np.float)
    elif type == 'ShearY':
        theta = np.array([[1, 0, 0], [v, 1, 0]]).astype(np.float)
    elif type == 'TranslateX':
        theta = np.array([[1, 0, v], [0, 1, 0]]).astype(np.float)
    elif type == 'TranslateY':
        theta = np.array([[1, 0, 0], [0, 1, v]]).astype(np.float)

    H = tensor.shape[2]
    W = tensor.shape[3]
    theta[0,1] = theta[0,1]*H/W
    theta[1,0] = theta[1,0]*W/H
    if type != 'Rotate':
        theta[0,2] = theta[0,2]*2/H + theta[0,0] + theta[0,1] - 1
        theta[1,2] = theta[1,2]*2/H + theta[1,0] + theta[1,1] - 1

    theta = torch.Tensor(theta).unsqueeze(0)
    grid = F.affine_grid(theta, tensor.size()).to(tensor.device)
    tensor_t = F.grid_sample(tensor, grid, mode='nearest')
    return tensor_t

if __name__ == '__main__':
    randaug = RandAugmentMC(2, 10)
    #path = r'E:\WorkHome\IMG_20190131_142431.jpg'
    path = r'E:\WorkHome\0.png'
    img = Image.open(path)
    img_t = transforms.ToTensor()(img).unsqueeze(0)
    #img_aug, aug_type = randaug(img)
    #img_aug.show()
    
    # v = 20
    # img_pil = img.rotate(v)
    # img_T = affine_sample(img_t, v, 'Rotate')

    v = 0.12
    img_pil = img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))
    img_T = affine_sample(img_t, v, 'ShearY')

    img_ten = transforms.ToPILImage()(img_T.squeeze(0))
    img_pil.show()
    img_ten.show()