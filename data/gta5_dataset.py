# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
import numpy as np
import scipy.misc as m
import matplotlib.pyplot as plt
import matplotlib.image  as imgs
from PIL import Image
import random
import scipy.io as io
from tqdm import tqdm
from scipy import stats

from torch.utils import data

from data import BaseDataset
from data.randaugment import RandAugmentMC

class GTA5_loader(BaseDataset):
    """
    GTA5    synthetic dataset
    for domain adaptation to Cityscapes
    """

    colors = [  # [  0,   0,   0],
        [128, 64, 128],
        [244, 35, 232],
        [70, 70, 70],
        [102, 102, 156],
        [190, 153, 153],
        [153, 153, 153],
        [250, 170, 30],
        [220, 220, 0],
        [107, 142, 35],
        [152, 251, 152],
        [0, 130, 180],
        [220, 20, 60],
        [255, 0, 0],
        [0, 0, 142],
        [0, 0, 70],
        [0, 60, 100],
        [0, 80, 100],
        [0, 0, 230],
        [119, 11, 32],
    ]

    label_colours = dict(zip(range(19), colors))
    def __init__(self, opt, logger, augmentations=None):
        self.opt = opt
        self.root = opt.src_rootpath
        self.split = 'all'
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = 19
        self.img_size = (1914, 1052)

        self.mean = [0.0, 0.0, 0.0] #TODO:  calculating the mean value of rgb channels on GTA5
        self.image_base_path = os.path.join(self.root, 'images')
        self.label_base_path = os.path.join(self.root, 'labels')
        splits = io.loadmat(os.path.join(self.root, 'split.mat'))
        if self.split == 'all':
            ids = np.concatenate((splits['trainIds'][:,0], splits['valIds'][:,0], splits['testIds'][:,0]))
        elif self.split == 'train':
            ids = splits['trainIds'][:,0]
        elif self.split == 'val':
            ids = splits['valIds'][:200,0]
        elif self.split == 'test':
            ids = splits['testIds'][:,0]
        self.ids = []
        for i in range(len(ids)):
            self.ids.append(os.path.join(self.label_base_path, str(i+1).zfill(5) + '.png'))


        self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, 34, -1]
        self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
        self.class_names = ["unlabelled","road","sidewalk","building","wall","fence","pole","traffic_light",
            "traffic_sign","vegetation","terrain","sky","person","rider","car","truck","bus","train",
            "motorcycle","bicycle",]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(19)))

        if len(self.ids) == 0:
            raise Exception(
                "No files for style=[%s] found in %s" % (self.split, self.image_base_path)
            )
        
        print("Found {} {} images".format(len(self.ids), self.split))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """__getitem__
        
        param: index
        """
        id = self.ids[index]
        if self.split != 'all' and self.split != 'val':
            filename = '{:05d}.png'.format(id)
            img_path = os.path.join(self.image_base_path, filename)
            lbl_path = os.path.join(self.label_base_path, filename)
        else:
            img_path = os.path.join(self.image_base_path, id.split('/')[-1])
            lbl_path = id
        
        img = Image.open(img_path)
        lbl = Image.open(lbl_path)

        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)
        lbl = np.asarray(lbl, dtype=np.uint8)

        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        input_dict = {}        
        if self.augmentations!=None:
            img, lbl, _, _, _ = self.augmentations(img, lbl)
            img_strong, params = self.randaug(Image.fromarray(img))
            img_strong, _ = self.transform(img_strong, lbl)
            input_dict['img_strong'] = img_strong
            input_dict['params'] = params

        img, lbl = self.transform(img, lbl)

        input_dict['img'] = img
        input_dict['label'] = lbl
        input_dict['img_path'] = self.ids[index]
        return input_dict


    def encode_segmap(self, lbl):
        for _i in self.void_classes:
            lbl[lbl == _i] = self.ignore_index
        for _i in self.valid_classes:
            lbl[lbl == _i] = self.class_map[_i]
        return lbl

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[l][0]
            g[temp == l] = self.label_colours[l][1]
            b[temp == l] = self.label_colours[l][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def transform(self, img, lbl):
        """transform

        img, lbl
        """
        img = np.array(img)
        # img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, self.img_size, "nearest", mode='F')
        lbl = lbl.astype(int)
        
        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes): 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl

    def get_cls_num_list(self):
        cls_num_list = np.array([16139327127,  4158369631,  8495419275,   927064742,   318109335,
                                532432540,    67453231,    40526481,  3818867486,  1081467674,
                                6800402117,   182228033,    15360044,  1265024472,   567736474,
                                184854135,    32542442,    15832619,     2721193])
        # cls_num_list = np.zeros(self.n_classes, dtype=np.int64)
        # for n in range(len(self.ids)):
        #     lbl = Image.open(self.ids[n])
        #     lbl = lbl.resize(self.img_size, Image.NEAREST)
        #     lbl = np.asarray(lbl, dtype=np.uint8)
        #     lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))
        #     for i in range(self.n_classes):
        #         cls_num_list[i] += (lbl == i).sum()
        return cls_num_list
