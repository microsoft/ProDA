# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import sys
import time
import torch
import random
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from parser_train import parser_, relative_path_to_absolute_path

from tqdm import tqdm
from data import create_dataset
from utils import get_logger
from models import adaptation_modelv2
from metrics import runningScore, averageMeter

def train(opt, logger):
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    np.random.seed(opt.seed)
    random.seed(opt.seed)
    ## create dataset
    device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
    datasets = create_dataset(opt, logger)

    if opt.model_name == 'deeplabv2':
        model = adaptation_modelv2.CustomModel(opt, logger)

    # Setup Metrics
    running_metrics_val = runningScore(opt.n_class)
    time_meter = averageMeter()

    # load category anchors
    if opt.stage == 'stage1':
        objective_vectors = torch.load(os.path.join(os.path.dirname(opt.resume_path), 'prototypes_on_{}_from_{}'.format(opt.tgt_dataset, opt.model_name)))
        model.objective_vectors = torch.Tensor(objective_vectors).to(0)

    # begin training
    save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name))
    model.iter = 0
    start_epoch = 0
    for epoch in range(start_epoch, opt.epochs):
        for data_i in datasets.target_train_loader:
            target_image = data_i['img'].to(device)
            target_imageS = data_i['img_strong'].to(device)
            target_params = data_i['params']
            target_image_full = data_i['img_full'].to(device)
            target_weak_params = data_i['weak_params']

            target_lp = data_i['lp'].to(device) if 'lp' in data_i.keys() else None
            target_lpsoft = data_i['lpsoft'].to(device) if 'lpsoft' in data_i.keys() else None
            source_data = datasets.source_train_loader.next()
            
            model.iter += 1
            i = model.iter
            images = source_data['img'].to(device)
            labels = source_data['label'].to(device)
            source_imageS = source_data['img_strong'].to(device)
            source_params = source_data['params']

            start_ts = time.time()

            model.train(logger=logger)
            if opt.freeze_bn:
                model.freeze_bn_apply()
            model.optimizer_zerograd()

            if opt.stage == 'warm_up':
                loss_GTA, loss_G, loss_D = model.step_adv(images, labels, target_image, source_imageS, source_params)
            elif opt.stage == 'stage1':
                loss, loss_CTS, loss_consist = model.step(images, labels, target_image, target_imageS, target_params, target_lp,
                                        target_lpsoft, target_image_full, target_weak_params)
            else:
                loss_GTA, loss = model.step_distillation(images, labels, target_image, target_imageS, target_params, target_lp)

            time_meter.update(time.time() - start_ts)

            #print(i)
            if (i + 1) % opt.print_interval == 0:
                if opt.stage == 'warm_up':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss_G: {:.4f}  loss_D: {:.4f} Time/Image: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_GTA, loss_G, loss_D, time_meter.avg / opt.bs)
                elif opt.stage == 'stage1':
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss: {:.4f}  loss_CTS: {:.4f}  loss_consist: {:.4f} Time/Image: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss, loss_CTS, loss_consist, time_meter.avg / opt.bs)
                else:
                    fmt_str = "Epochs [{:d}/{:d}] Iter [{:d}/{:d}]  loss_GTA: {:.4f}  loss: {:.4f} Time/Image: {:.4f}"
                    print_str = fmt_str.format(epoch+1, opt.epochs, i + 1, opt.train_iters, loss_GTA, loss, time_meter.avg / opt.bs)
                print(print_str)
                logger.info(print_str)
                time_meter.reset()

            # evaluation
            if (i + 1) % opt.val_interval == 0:
                validation(model, logger, datasets, device, running_metrics_val, iters = model.iter, opt=opt)
                torch.cuda.empty_cache()
                logger.info('Best iou until now is {}'.format(model.best_iou))

            model.scheduler_step()

def validation(model, logger, datasets, device, running_metrics_val, iters, opt=None):
    iters = iters
    _k = -1
    for v in model.optimizers:
        _k += 1
        for param_group in v.param_groups:
            _learning_rate = param_group.get('lr')
        logger.info("learning rate is {} for {} net".format(_learning_rate, model.nets[_k].__class__.__name__))
    model.eval(logger=logger)
    torch.cuda.empty_cache()
    val_datset = datasets.target_valid_loader
    #val_datset = datasets.target_train_loader
    with torch.no_grad():
        validate(val_datset, device, model, running_metrics_val)

    score, class_iou = running_metrics_val.get_scores()
    for k, v in score.items():
        print(k, v)
        logger.info('{}: {}'.format(k, v))

    for k, v in class_iou.items():
        logger.info('{}: {}'.format(k, v))

    running_metrics_val.reset()

    torch.cuda.empty_cache()
    state = {}
    _k = -1
    for net in model.nets:
        _k += 1
        new_state = {
            "model_state": net.state_dict(),
            #"optimizer_state": model.optimizers[_k].state_dict(),
            #"scheduler_state": model.schedulers[_k].state_dict(),  
            "objective_vectors": model.objective_vectors,
        }
        state[net.__class__.__name__] = new_state
    state['iter'] = iters + 1
    state['best_iou'] = score["Mean IoU : \t"]
    save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_current_model.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name))
    torch.save(state, save_path)

    if score["Mean IoU : \t"] >= model.best_iou:
        torch.cuda.empty_cache()
        model.best_iou = score["Mean IoU : \t"]
        state = {}
        _k = -1
        for net in model.nets:
            _k += 1
            new_state = {
                "model_state": net.state_dict(),
                #"optimizer_state": model.optimizers[_k].state_dict(),
                #"scheduler_state": model.schedulers[_k].state_dict(),     
                "objective_vectors": model.objective_vectors,                
            }
            state[net.__class__.__name__] = new_state
        state['iter'] = iters + 1
        state['best_iou'] = model.best_iou
        save_path = os.path.join(opt.logdir,"from_{}_to_{}_on_{}_best_model.pkl".format(opt.src_dataset, opt.tgt_dataset, opt.model_name))
        torch.save(state, save_path)
        return score["Mean IoU : \t"]

def validate(valid_loader, device, model, running_metrics_val):
    for data_i in tqdm(valid_loader):

        images_val = data_i['img'].to(device)
        labels_val = data_i['label'].to(device)

        out = model.BaseNet_DP(images_val)

        outputs = F.interpolate(out['out'], size=images_val.size()[2:], mode='bilinear', align_corners=True)
        #val_loss = loss_fn(input=outputs, target=labels_val)

        pred = outputs.data.max(1)[1].cpu().numpy()
        gt = labels_val.data.cpu().numpy()
        running_metrics_val.update(gt, pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="config")
    parser = parser_(parser)
    opt = parser.parse_args()

    opt = relative_path_to_absolute_path(opt)

    print('RUNDIR: {}'.format(opt.logdir))
    if not os.path.exists(opt.logdir):
        os.makedirs(opt.logdir)

    logger = get_logger(opt.logdir)

    train(opt, logger)
