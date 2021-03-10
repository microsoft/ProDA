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
from utils import fliplr

def test(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger) 

    if opt.model_name == 'deeplabv2':
        checkpoint = torch.load(opt.resume_path)['ResNet101']["model_state"]
        model = adaptation_modelv2.CustomModel(opt, logger)
        model.BaseNet.load_state_dict(checkpoint)

    validation(model, logger, datasets, device, opt)

def validation(model, logger, datasets, device, opt):
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(datasets.target_train_loader, device, model, opt)
        #validate(datasets.target_valid_loader, device, model, opt)

def label2rgb(func, label):
    rgbs = []
    for k in range(label.shape[0]):
        rgb = func(label[k, 0].cpu().numpy())
        rgbs.append(torch.from_numpy(rgb).permute(2, 0, 1))
    rgbs = torch.stack(rgbs, dim=0).float()
    return rgbs

def validate(valid_loader, device, model, opt):
    ori_LP = os.path.join(opt.root, 'Code/ProDA', opt.save_path, opt.name)

    if not os.path.exists(ori_LP):
        os.makedirs(ori_LP)

    sm = torch.nn.Softmax(dim=1)
    for data_i in tqdm(valid_loader):
        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)
        filename = data_i['img_path']

        out = model.BaseNet_DP(images_val)

        if opt.soft:
            threshold_arg = F.softmax(out['out'], dim=1)
            for k in range(labels_val.shape[0]):
                name = os.path.basename(filename[k])
                np.save(os.path.join(ori_LP, name.replace('.png', '.npy')), threshold_arg[k].cpu().numpy())
        else:
            if opt.flip:
                flip_out = model.BaseNet_DP(fliplr(images_val))
                flip_out['out'] = F.interpolate(sm(flip_out['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
                out['out'] = F.interpolate(sm(out['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
                out['out'] = (out['out'] + fliplr(flip_out['out'])) / 2

            confidence, pseudo = out['out'].max(1, keepdim=True)
            #entropy = -(out['out']*torch.log(out['out']+1e-6)).sum(1, keepdim=True)
            pseudo_rgb = label2rgb(valid_loader.dataset.decode_segmap, pseudo).float() * 255
            for k in range(labels_val.shape[0]):
                name = os.path.basename(filename[k])
                Image.fromarray(pseudo[k,0].cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name))
                Image.fromarray(pseudo_rgb[k].permute(1,2,0).cpu().numpy().astype(np.uint8)).save(os.path.join(ori_LP, name[:-4] + '_color.png'))
                np.save(os.path.join(ori_LP, name.replace('.png', '_conf.npy')), confidence[k, 0].cpu().numpy().astype(np.float16))
                #np.save(os.path.join(ori_LP, name.replace('.png', '_entropy.npy')), entropy[k, 0].cpu().numpy().astype(np.float16))
                
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
    parser.add_argument('--save_path', type=str, default='./Pseudo', help='pseudo label update thred')
    parser.add_argument('--soft', action='store_true', help='save soft pseudo label')
    parser.add_argument('--flip', action='store_true')
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')
    opt.noaug = True
    opt.noshuffle = True

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)

#python generate_pseudo_label.py --name gta2citylabv2_warmup_soft --soft --resume_path ./logs/gta2citylabv2_warmup/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage1Denoise --flip --resume_path ./logs/gta2citylabv2_stage1Denoisev2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast
#python generate_pseudo_label.py --name gta2citylabv2_stage2 --flip --resume_path ./logs/gta2citylabv2_stage2/from_gta5_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast --bn_clr --student_init simclr
#python generate_pseudo_label.py --name syn2citylabv2_warmup_soft --soft --src_dataset synthia --n_class 16 --src_rootpath Dataset/SYNTHIA-RAND-CITYSCAPES --resume_path ./logs/syn2citylabv2_warmup/from_synthia_to_cityscapes_on_deeplabv2_best_model.pkl --no_droplast