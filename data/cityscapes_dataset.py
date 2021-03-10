# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import torch
import numpy as np
import scipy.misc as m
from tqdm import tqdm

from torch.utils import data
from PIL import Image

from data.augmentations import *
from data.base_dataset import BaseDataset
from data.randaugment import RandAugmentMC

import random

def recursive_glob(rootdir=".", suffix=""):
    """Performs recursive glob with given suffix and rootdir 
        :param rootdir is the root directory
        :param suffix is the suffix to be searched
    """
    return [
        os.path.join(looproot, filename)
        for looproot, _, filenames in os.walk(rootdir)  #os.walk: traversal all files in rootdir and its subfolders
        for filename in filenames
        if filename.endswith(suffix)
    ]

class Cityscapes_loader(BaseDataset):
    """cityscapesLoader

    https://www.cityscapes-dataset.com

    Data is derived from CityScapes, and can be downloaded from here:
    https://www.cityscapes-dataset.com/downloads/

    Many Thanks to @fvisin for the loader repo:
    https://github.com/fvisin/dataset_loaders/blob/master/dataset_loaders/images/cityscapes.py
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

    mean_rgb = {
        "pascal": [103.939, 116.779, 123.68],
        "cityscapes": [0.0, 0.0, 0.0],
    }  # pascal mean for PSPNet and ICNet pre-trained model

    def __init__(self, opt, logger, augmentations = None, split='train'):
        """__init__

        :param opt: parameters of dataset
        :param writer: save the result of experiment
        :param logger: logging file
        :param augmentations: 
        """
        
        self.opt = opt
        self.root = opt.tgt_rootpath
        self.split = split
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = opt.n_class
        self.img_size = (2048, 1024)
        self.mean = np.array(self.mean_rgb['cityscapes'])
        self.files = {}
        self.paired_files = {}

        self.images_base = os.path.join(self.root, "leftImg8bit", self.split)
        self.annotations_base = os.path.join(
            self.root, "gtFine", self.split
        )

        self.files = sorted(recursive_glob(rootdir=self.images_base, suffix=".png")) #find all files from rootdir and subfolders with suffix = ".png"

        #self.void_classes = [0, 1, 2, 3, 4, 5, 6, 9, 10, 14, 15, 16, 18, 29, 30, -1]
        if self.n_classes == 19:
            self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 31, 32, 33,]
            self.class_names = ["unlabelled","road","sidewalk","building","wall",
                "fence","pole","traffic_light","traffic_sign","vegetation",
                "terrain","sky","person","rider","car",
                "truck","bus","train","motorcycle","bicycle",
            ]
            self.to19 = dict(zip(range(19), range(19)))
        elif self.n_classes == 16:
            self.valid_classes = [7, 8, 11, 12, 13, 17, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33,]
            self.class_names = ["unlabelled","road","sidewalk","building","wall",
                "fence","pole","traffic_light","traffic_sign","vegetation",
                "sky","person","rider","car","bus",
                "motorcycle","bicycle",
            ]
            self.to19 = dict(zip(range(16), [0,1,2,3,4,5,6,7,8,10,11,12,13,15,17,18]))
        elif self.n_classes == 13:
            self.valid_classes = [7, 8, 11, 19, 20, 21, 23, 24, 25, 26, 28, 32, 33,]
            self.class_names = ["unlabelled","road","sidewalk","building","traffic_light",
                "traffic_sign","vegetation","sky","person","rider",
                "car","bus","motorcycle","bicycle",
            ]
            self.to19 = dict(zip(range(13), [0,1,2,6,7,8,10,11,12,13,15,17,18]))

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))   #zip: return tuples

        if not self.files:
            raise Exception(
                "No files for split=[%s] found in %s" % (self.split, self.images_base)
            )

        print("Found %d %s images" % (len(self.files), self.split))
    
    def __len__(self):
        """__len__"""
        return len(self.files)

    def __getitem__(self, index):
        """__getitem__

        :param index:
        """
        img_path = self.files[index].rstrip()
        lbl_path = os.path.join(
            self.annotations_base,
            img_path.split(os.sep)[-2],
            os.path.basename(img_path)[:-15] + "gtFine_labelIds.png",
        )

        img = Image.open(img_path)
        lbl = Image.open(lbl_path)
        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        
        img = np.array(img, dtype=np.uint8)
        lbl = np.array(lbl, dtype=np.uint8)
        lbl = self.encode_segmap(np.array(lbl, dtype=np.uint8))

        img_full = img.copy().astype(np.float64)
        img_full -= self.mean
        img_full = img_full.astype(float) / 255.0
        img_full = img_full.transpose(2, 0, 1)

        lp, lpsoft, weak_params = None, None, None
        if self.split == 'train' and self.opt.used_save_pseudo:
            if self.opt.proto_rectify:
                lpsoft = np.load(os.path.join(self.opt.path_soft, os.path.basename(img_path).replace('.png', '.npy')))
            else:
                lp_path = os.path.join(self.opt.path_LP, os.path.basename(img_path))
                lp = Image.open(lp_path)
                lp = lp.resize(self.img_size, Image.NEAREST)
                lp = np.array(lp, dtype=np.uint8)
                if self.opt.threshold:
                    conf = np.load(os.path.join(self.opt.path_LP, os.path.basename(img_path).replace('.png', '_conf.npy')))
                    lp[conf <= self.opt.threshold] = 250

        input_dict = {}
        if self.augmentations!=None:
            img, lbl, lp, lpsoft, weak_params = self.augmentations(img, lbl, lp, lpsoft)
            img_strong, params = self.randaug(Image.fromarray(img))
            img_strong, _, _ = self.transform(img_strong, lbl)
            input_dict['img_strong'] = img_strong
            input_dict['params'] = params

        img, lbl_, lp = self.transform(img, lbl, lp)
                
        input_dict['img'] = img
        input_dict['img_full'] = torch.from_numpy(img_full).float()
        input_dict['label'] = lbl_
        input_dict['lp'] = lp
        input_dict['lpsoft'] = lpsoft
        input_dict['weak_params'] = weak_params  #full2weak
        input_dict['img_path'] = self.files[index]

        input_dict = {k:v for k, v in input_dict.items() if v is not None}
        return input_dict

    def transform(self, img, lbl, lp=None, check=True):
        """transform

        :param img:
        :param lbl:
        """
        # img = m.imresize(
        #     img, (self.img_size[0], self.img_size[1])
        # )  # uint8 with RGB mode
        img = np.array(img)
        # img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean
        img = img.astype(float) / 255.0
        # NHWC -> NCHW
        img = img.transpose(2, 0, 1)

        classes = np.unique(lbl)
        lbl = np.array(lbl)
        lbl = lbl.astype(float)
        # lbl = m.imresize(lbl, (self.img_size[0], self.img_size[1]), "nearest", mode="F")
        lbl = lbl.astype(int)

        if not np.all(classes == np.unique(lbl)):
            print("WARN: resizing labels yielded fewer classes")    #TODO: compare the original and processed ones

        if check and not np.all(np.unique(lbl[lbl != self.ignore_index]) < self.n_classes):   #todo: understanding the meaning 
            print("after det", classes, np.unique(lbl))
            raise ValueError("Segmentation map contained invalid class values")

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        if lp is not None:
            classes = np.unique(lp)
            lp = np.array(lp)
            # if not np.all(np.unique(lp[lp != self.ignore_index]) < self.n_classes):
            #     raise ValueError("lp Segmentation map contained invalid class values")
        
            lp = torch.from_numpy(lp).long()

        return img, lbl, lp

    def decode_segmap(self, temp):
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.n_classes):
            r[temp == l] = self.label_colours[self.to19[l]][0]
            g[temp == l] = self.label_colours[self.to19[l]][1]
            b[temp == l] = self.label_colours[self.to19[l]][2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r / 255.0
        rgb[:, :, 1] = g / 255.0
        rgb[:, :, 2] = b / 255.0
        return rgb

    def encode_segmap(self, mask):
        # Put all void classes to zero
        label_copy = 250 * np.ones(mask.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[mask == k] = v
        return label_copy

    def get_cls_num_list(self):
        cls_num_list = np.array([1557726944,  254364912,  673500400,   18431664,   14431392,
                                29361440,    7038112,    7352368,  477239920,   40134240,
                                211669120,   36057968,     865184,  264786464,   17128544,
                                2385680,     943312,     504112,    2174560])
        return cls_num_list
