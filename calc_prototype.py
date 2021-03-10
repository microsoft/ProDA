# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from parser_train import parser_, relative_path_to_absolute_path
from tqdm import tqdm

from data import create_dataset
from models import adaptation_modelv2

def calc_prototype(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger) 

    if opt.model_name == 'deeplabv2':
        model = adaptation_modelv2.CustomModel(opt, logger)

    class_features = Class_Features(numbers=opt.n_class)

    # begin training
    model.iter = 0
    for epoch in range(opt.epochs):
        for data_i in tqdm(datasets.target_train_loader):  
            model.iter += 1
            i = model.iter
            source_data = datasets.source_train_loader.next()
            images = source_data['img'].to(device)
            labels = source_data['label'].to(device)

            target_image = data_i['img'].to(device)
            target_label = data_i['label'].to(device)

            model.eval()
            if opt.source: #source
                with torch.no_grad():
                    if opt.model_name == 'deeplabv2':
                        out = model.BaseNet_DP(images, ssl=True)
                    batch, w, h = labels.size()
                    newlabels = labels.reshape([batch, 1, w, h]).float()
                    newlabels = F.interpolate(newlabels, size=out['feat'].size()[2:], mode='nearest')
                    vectors, ids = class_features.calculate_mean_vector(out['feat'], out['out'], newlabels, model)
                    for t in range(len(ids)):
                        model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu().numpy(), 'mean')
            else: #target
                with torch.no_grad():
                    if opt.model_name == 'deeplabv2':
                        out = model.BaseNet_DP(target_image, ssl=True)
                    vectors, ids = class_features.calculate_mean_vector(out['feat'], out['out'], model=model)
                    #vectors, ids = class_features.calculate_mean_vector_by_output(feat_cls, output, model)
                    for t in range(len(ids)):
                        model.update_objective_SingleVector(ids[t], vectors[t].detach().cpu(), 'mean')

    if opt.source:
        save_path = os.path.join(os.path.dirname(opt.resume_path), "prototypes_on_{}_from_{}".format(opt.src_dataset, opt.model_name))
    else:
        save_path = os.path.join(os.path.dirname(opt.resume_path), "prototypes_on_{}_from_{}".format(opt.tgt_dataset, opt.model_name))
    torch.save(model.objective_vectors, save_path)

class Class_Features:
    def __init__(self, numbers = 19):
        self.class_numbers = numbers
        self.class_features = [[] for i in range(self.class_numbers)]
        self.num = np.zeros(numbers)

    def calculate_mean_vector_by_output(self, feat_cls, outputs, model):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())
        outputs_pred = outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def calculate_mean_vector(self, feat_cls, outputs, labels_val=None, model=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = model.process_label(outputs_argmax.float())
        if labels_val is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = model.process_label(labels_val)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t]
                # if (torch.sum(outputs_pred[n][t] * labels_expanded[n][t]).item() < 30):
                #     continue
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                # self.update_cls_feature(vector=s, id=t)
                vectors.append(s)
                ids.append(t)
        return vectors, ids

def get_logger(logdir):
    logger = logging.getLogger('ptsemseg')
    file_path = os.path.join(logdir, 'run.log')
    hdlr = logging.FileHandler(file_path)
    formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    hdlr.setFormatter(formatter)
    logger.addHandler(hdlr) 
    logger.setLevel(logging.INFO)
    return logger

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser.add_argument('--source', action='store_true', help='calc source prototype')
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')
    opt.noaug = True
    opt.noshuffle = True
    opt.epochs = 4
    #opt.num_workers = 0

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    calc_prototype(opt, logger)

#python calc_prototype.py --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl
#python calc_prototype.py --resume_path ./logs/syn2citylabv2_warmup/from_synthia_to_cityscapes_on_deeplabv2_best_model.pkl --src_dataset synthia --n_class 16 --src_rootpath Dataset/SYNTHIA-RAND-CITYSCAPES