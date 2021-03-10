# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import logging
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from parser_train import parser_, relative_path_to_absolute_path
from tqdm import tqdm

from data import create_dataset
from models import adaptation_modelv2
from metrics import runningScore

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
    
    running_metrics_val = runningScore(opt.n_class)

    validation(model, logger, datasets, device, running_metrics_val)

def validation(model, logger, datasets, device, running_metrics_val):
    _k = -1
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    with torch.no_grad():
        validate(datasets.target_valid_loader, device, model, running_metrics_val)
        
    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))

    running_metrics_val.reset()

    torch.cuda.empty_cache()
    return score["Mean IoU : \t"]

def validate(valid_loader, device, model, running_metrics_val):
    sm = torch.nn.Softmax(dim=1)
    for data_i in tqdm(valid_loader):
        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)

        outs = model.BaseNet_DP(images_val)
        #outputs = F.interpolate(sm(outs['out']), size=images_val.size()[2:], mode='bilinear', align_corners=True)
        outputs = F.interpolate(outs['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

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
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)
    opt.logdir = opt.logdir.replace(opt.name, 'debug')

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    test(opt, logger)


