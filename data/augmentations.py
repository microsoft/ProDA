# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
# Adapted from https://github.com/ZijunDeng/pytorch-semantic-segmentation/blob/master/utils/joint_transforms.py

import math
import numbers
import random
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tf

from PIL import Image, ImageOps


class Compose(object):
    def __init__(self, augmentations):
        self.augmentations = augmentations
        self.PIL2Numpy = False

    def __call__(self, img, mask, mask1=None, lpsoft=None):
        params = {}
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img, mode="RGB")
            mask = Image.fromarray(mask, mode="L")
            if mask1 is not None:
                mask1 = Image.fromarray(mask1, mode="L")
            if lpsoft is not None:
                lpsoft = torch.from_numpy(lpsoft)
                lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[img.size[1], img.size[0]], mode='bilinear', align_corners=True)[0]
            self.PIL2Numpy = True

        if img.size != mask.size:
            print (img.size, mask.size)
        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)
        for a in self.augmentations:
            img, mask, mask1, lpsoft, params = a(img, mask, mask1, lpsoft, params)
            # print(img.size)

        if self.PIL2Numpy:
            img, mask = np.array(img), np.array(mask, dtype=np.uint8) 
            if mask1 is not None:
                mask1 = np.array(mask1, dtype=np.uint8)
        return img, mask, mask1, lpsoft, params


class RandomCrop(object):
    def __init__(self, size, padding=0):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size
        self.padding = padding

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        if self.padding > 0:
            img = ImageOps.expand(img, border=self.padding, fill=0)
            mask = ImageOps.expand(mask, border=self.padding, fill=0)
            if mask1 is not None:
                mask1 = ImageOps.expand(mask1, border=self.padding, fill=0)

        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)
        w, h = img.size
        tw, th = self.size
        if w == tw and h == th:
            return img, mask
        if w < tw or h < th:
            if lpsoft is not None:
                lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[th, tw], mode='bolinear', align_corners=True)[0]
            if mask1 is not None:
                return (
                        img.resize((tw, th), Image.BILINEAR),
                        mask.resize((tw, th), Image.NEAREST),
                        mask1.resize((tw, th), Image.NEAREST),
                        lpsoft
                    )
            else:
                    return (
                        img.resize((tw, th), Image.BILINEAR),
                        mask.resize((tw, th), Image.NEAREST),
                        None,
                        lpsoft
                    )

        x1 = random.randint(0, w - tw)
        y1 = random.randint(0, h - th)
        params['RandomCrop'] = (y1, y1 + th, x1, x1 + tw)
        if lpsoft is not None:
            lpsoft = lpsoft[:, y1:y1 + th, x1:x1 + tw]
        if mask1 is not None:
            return (
                img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                mask1.crop((x1, y1, x1 + tw, y1 + th)),
                lpsoft,
                params
            )
        else:
            return (
                img.crop((x1, y1, x1 + tw, y1 + th)),
                mask.crop((x1, y1, x1 + tw, y1 + th)),
                None,
                lpsoft,
                params
            )


class AdjustGamma(object):
    def __init__(self, gamma):
        self.gamma = gamma

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_gamma(img, random.uniform(1, 1 + self.gamma)), mask


class AdjustSaturation(object):
    def __init__(self, saturation):
        self.saturation = saturation

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_saturation(img, 
                                    random.uniform(1 - self.saturation, 
                                                   1 + self.saturation)), mask


class AdjustHue(object):
    def __init__(self, hue):
        self.hue = hue

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_hue(img, random.uniform(-self.hue, 
                                                  self.hue)), mask


class AdjustBrightness(object):
    def __init__(self, bf):
        self.bf = bf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_brightness(img, 
                                    random.uniform(1 - self.bf, 
                                                   1 + self.bf)), mask

class AdjustContrast(object):
    def __init__(self, cf):
        self.cf = cf

    def __call__(self, img, mask):
        assert img.size == mask.size
        return tf.adjust_contrast(img, 
                                  random.uniform(1 - self.cf, 
                                                 1 + self.cf)), mask

class CenterCrop(object):
    def __init__(self, size):
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        th, tw = self.size
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return (
            img.crop((x1, y1, x1 + tw, y1 + th)),
            mask.crop((x1, y1, x1 + tw, y1 + th)),
        )


class RandomHorizontallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        if random.random() < self.p:
            params['RandomHorizontallyFlip'] = True
            if lpsoft is not None:
                inv_idx = torch.arange(lpsoft.size(2)-1,-1,-1).long()  # C x H x W
                lpsoft = lpsoft.index_select(2,inv_idx)
            if mask1 is not None:
                return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    mask1.transpose(Image.FLIP_LEFT_RIGHT),
                    lpsoft,
                    params
                )
            else:
                return (
                    img.transpose(Image.FLIP_LEFT_RIGHT),
                    mask.transpose(Image.FLIP_LEFT_RIGHT),
                    None,
                    lpsoft,
                    params
                )
        else:
            params['RandomHorizontallyFlip'] = False
        return img, mask, mask1, lpsoft, params


class RandomVerticallyFlip(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img, mask):
        if random.random() < self.p:
            return (
                img.transpose(Image.FLIP_TOP_BOTTOM),
                mask.transpose(Image.FLIP_TOP_BOTTOM),
            )
        return img, mask


class FreeScale(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask):
        assert img.size == mask.size
        return (
            img.resize(self.size, Image.BILINEAR),
            mask.resize(self.size, Image.NEAREST),
        )


class RandomTranslate(object):
    def __init__(self, offset):
        self.offset = offset # tuple (delta_x, delta_y)

    def __call__(self, img, mask):
        assert img.size == mask.size
        x_offset = int(2 * (random.random() - 0.5) * self.offset[0])
        y_offset = int(2 * (random.random() - 0.5) * self.offset[1])
        
        x_crop_offset = x_offset
        y_crop_offset = y_offset
        if x_offset < 0:
            x_crop_offset = 0
        if y_offset < 0:
            y_crop_offset = 0
        
        cropped_img = tf.crop(img, 
                              y_crop_offset, 
                              x_crop_offset, 
                              img.size[1]-abs(y_offset), 
                              img.size[0]-abs(x_offset))

        if x_offset >= 0 and y_offset >= 0:
            padding_tuple = (0, 0, x_offset, y_offset)

        elif x_offset >= 0 and y_offset < 0:
            padding_tuple = (0, abs(y_offset), x_offset, 0)
        
        elif x_offset < 0 and y_offset >= 0:
            padding_tuple = (abs(x_offset), 0, 0, y_offset)
        
        elif x_offset < 0 and y_offset < 0:
            padding_tuple = (abs(x_offset), abs(y_offset), 0, 0)
        
        return (
              tf.pad(cropped_img, 
                     padding_tuple, 
                     padding_mode='reflect'),
              tf.affine(mask,
                        translate=(-x_offset, -y_offset),
                        scale=1.0,
                        angle=0.0,
                        shear=0.0,
                        fillcolor=250))


class RandomRotate(object):
    def __init__(self, degree):
        self.degree = degree

    def __call__(self, img, mask):
        rotate_degree = random.random() * 2 * self.degree - self.degree
        return (
            tf.affine(img, 
                      translate=(0, 0),
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.BILINEAR,
                      fillcolor=(0, 0, 0),
                      shear=0.0),
            tf.affine(mask, 
                      translate=(0, 0), 
                      scale=1.0, 
                      angle=rotate_degree, 
                      resample=Image.NEAREST,
                      fillcolor=250,
                      shear=0.0))



class Scale(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        w, h = img.size
        if (w >= h and w == self.size) or (h >= w and h == self.size):
            return img, mask
        if w > h:
            ow = self.size
            oh = int(self.size * h / w)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )
        else:
            oh = self.size
            ow = int(self.size * w / h)
            return (
                img.resize((ow, oh), Image.BILINEAR),
                mask.resize((ow, oh), Image.NEAREST),
            )

def MyScale(img, lbl, size):
    """scale

    img, lbl, longer size
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        _lbl = Image.fromarray(lbl)
    else:
        _img = img
        _lbl = lbl
    assert _img.size == _lbl.size
    # prop = 1.0 * _img.size[0]/_img.size[1]
    w, h = size
    # h = int(size / prop)
    _img = _img.resize((w, h), Image.BILINEAR)
    _lbl = _lbl.resize((w, h), Image.NEAREST)
    return np.array(_img), np.array(_lbl)

def Flip(img, lbl, prop):
    """
    flip img and lbl with probablity prop
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        _lbl = Image.fromarray(lbl)
    else:
        _img = img
        _lbl = lbl
    if random.random() < prop:
        _img.transpose(Image.FLIP_LEFT_RIGHT),
        _lbl.transpose(Image.FLIP_LEFT_RIGHT),
    return np.array(_img), np.array(_lbl)

def MyRotate(img, lbl, degree):
    """
    img, lbl, degree
    randomly rotate clockwise or anti-clockwise
    """
    if isinstance(img, np.ndarray):
        _img = Image.fromarray(img)
        _lbl = Image.fromarray(lbl)
    else:
        _img = img
        _lbl = lbl
    _degree = random.random()*degree
    
    flags = -1
    if random.random() < 0.5:
        flags = 1
    _img = _img.rotate(_degree * flags)
    _lbl = _lbl.rotate(_degree * flags)
    return np.array(_img), np.array(_lbl)

class RandomSizedCrop(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, img, mask):
        assert img.size == mask.size
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(0.45, 1.0) * area
            aspect_ratio = random.uniform(0.5, 2)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                mask = mask.crop((x1, y1, x1 + w, y1 + h))
                assert img.size == (w, h)

                return (
                    img.resize((self.size, self.size), Image.BILINEAR),
                    mask.resize((self.size, self.size), Image.NEAREST),
                )

        # Fallback
        scale = Scale(self.size)
        crop = CenterCrop(self.size)
        return crop(*scale(img, mask))


class RandomSized(object):
    def __init__(self, size):
        self.size = size
        self.scale = Scale(self.size)
        self.crop = RandomCrop(self.size)

    def __call__(self, img, mask, mask1=None, lpsoft=None, params=None):
        assert img.size == mask.size
        if mask1 is not None:
            assert (img.size == mask1.size)

        prop = 1.0 * img.size[0] / img.size[1]
        w = int(random.uniform(0.5, 1.5) * self.size)
        #w = self.size
        h = int(w/prop)
        params['RandomSized'] = (h, w)
        # h = int(random.uniform(0.5, 2) * self.size[1])

        img, mask = (
            img.resize((w, h), Image.BILINEAR),
            mask.resize((w, h), Image.NEAREST),
        )
        if mask1 is not None:
            mask1 = mask1.resize((w, h), Image.NEAREST)
        if lpsoft is not None:
            lpsoft = F.interpolate(lpsoft.unsqueeze(0), size=[h, w], mode='bilinear', align_corners=True)[0]

        return img, mask, mask1, lpsoft, params
        # return self.crop(*self.scale(img, mask))
