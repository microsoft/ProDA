# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import torch
import numpy as np
from PIL import Image
import random
import imageio

from data import BaseDataset
from data.randaugment import RandAugmentMC

class Synthia_loader(BaseDataset):
    """
    Synthia    synthetic dataset
    for domain adaptation to Cityscapes
    """

    def __init__(self, opt, logger, augmentations=None):
        self.opt = opt
        self.root = opt.src_rootpath
        self.augmentations = augmentations
        self.randaug = RandAugmentMC(2, 10)
        self.n_classes = opt.n_class
        self.img_size = (1280, 760)

        self.mean = [0.0, 0.0, 0.0] #TODO:  calculating the mean value of rgb channels on GTA5
        self.image_base_path = os.path.join(self.root, 'RGB')
        self.label_base_path = os.path.join(self.root, 'GT/LABELS')
        self.distribute = np.zeros(self.n_classes, dtype=float)
        ids = os.listdir(self.image_base_path)
        self.ids = []
        for i in range(len(ids)):
            self.ids.append(os.path.join(self.label_base_path, ids[i]))

        if self.n_classes == 19:
            self.valid_classes = [3,4,2,21,5,7,15,9,6,16,1,10,17,8,18,19,20,12,11,]
            self.class_names = ["unlabelled","Road","Sidewalk","Building","Wall",
                "Fence","Pole","Traffic_light","Traffic_sign","Vegetation",
                "Terrain","sky","Pedestrian","Rider","Car",
                "Truck","Bus","Train","Motorcycle","Bicycle",
            ]
        elif self.n_classes == 16:
            self.valid_classes = [3,4,2,21,5,7,15,9,6,1,10,17,8,19,12,11,]
            self.class_names = ["unlabelled","Road","Sidewalk","Building","Wall",
                "Fence","Pole","Traffic_light","Traffic_sign","Vegetation",
                "sky","Pedestrian","Rider","Car","Bus",
                "Motorcycle","Bicycle",
            ]
        elif self.n_classes == 13:
            self.valid_classes = [3,4,2,15,9,6,1,10,17,8,19,12,11,]
            self.class_names = ["unlabelled","Road","Sidewalk","Building","Traffic_light",
                "Traffic_sign","Vegetation","sky","Pedestrian","Rider",
                "Car","Bus","Motorcycle","Bicycle",
            ]

        self.ignore_index = 250
        self.class_map = dict(zip(self.valid_classes, range(self.n_classes)))
        imageio.plugins.freeimage.download()

        if len(self.ids) == 0:
            raise Exception(
                "No files found in %s" % (self.image_base_path)
            )
        
        print("Found {} images".format(len(self.ids)))

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        """__getitem__
        
        param: index
        """
        id = self.ids[index]

        img_path = os.path.join(self.image_base_path, id.split('/')[-1])
        lbl_path = id
        
        img = Image.open(img_path)
        lbl = np.asarray(imageio.imread(lbl_path, format='PNG-FI'))[:,:,0]
        lbl = Image.fromarray(lbl)

        img = img.resize(self.img_size, Image.BILINEAR)
        lbl = lbl.resize(self.img_size, Image.NEAREST)
        img = np.asarray(img, dtype=np.uint8)
        # lbl = lbl.convert('L')
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
        label_copy = 250 * np.ones(lbl.shape, dtype=np.uint8)
        for k, v in list(self.class_map.items()):
            label_copy[lbl == k] = v
        return label_copy

    # def decode_segmap(self, temp):
    #     r = temp.copy()
    #     g = temp.copy()
    #     b = temp.copy()
    #     for l in range(0, self.n_classes):
    #         r[temp == l] = self.label_colours[l][0]
    #         g[temp == l] = self.label_colours[l][1]
    #         b[temp == l] = self.label_colours[l][2]

    #     rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
    #     rgb[:, :, 0] = r / 255.0
    #     rgb[:, :, 1] = g / 255.0
    #     rgb[:, :, 2] = b / 255.0
    #     return rgb

    def transform(self, img, lbl):
        """transform

        img, lbl
        """
        # img = m.imresize(
        #     img, self.img_size,
        # )
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
        return None
