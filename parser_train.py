# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os
import json

def parser_(parser):
    parser.add_argument('--root', type=str, default='/mnt/blob', help='root path')
    parser.add_argument('--model_name', type=str, default='deeplabv2', help='deeplabv2')
    parser.add_argument('--name', type=str, default='gta2city', help='pretrain source model')
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--bs', type=int, default=4)
    parser.add_argument('--freeze_bn', action='store_true')
    parser.add_argument('--epochs', type=int, default=84)
    parser.add_argument('--train_iters', type=int, default=90000)
    parser.add_argument('--moving_prototype', action='store_true')
    parser.add_argument('--bn', type=str, default='sync_bn', help='sync_bn|bn|gn|adabn')
    #training
    parser.add_argument('--no_resume', action='store_true')
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--stage', type=str, default='stage1', help='warm_up|stage1|stage2|stage3')
    parser.add_argument('--finetune', action='store_true')
    #model
    parser.add_argument('--resume_path', type=str, default='pretrained/warmup/from_gta5_to_cityscapes_on_deeplab101_best_model_warmup.pkl', help='resume model path')
    parser.add_argument('--ema', action='store_true', help='use ema model')
    parser.add_argument('--ema_bn', action='store_true', help='add extra bn for ema model')
    parser.add_argument("--student_init", default='stage1', type=str, help="stage1|imagenet|simclr")
    parser.add_argument("--proto_momentum", default=0.0001, type=float)
    parser.add_argument("--bn_clr", action='store_true', help="if true, add a bn layer for the output of simclr model")
    #data
    parser.add_argument('--src_dataset', type=str, default='gta5', help='gta5|synthia')
    parser.add_argument('--tgt_dataset', type=str, default='cityscapes', help='cityscapes')
    parser.add_argument('--src_rootpath', type=str, default='Dataset/GTA5')
    parser.add_argument('--tgt_rootpath', type=str, default='Dataset/cityscapes')
    parser.add_argument('--path_LP', type=str, default='Pseudo/pretrain_warmup/LP0.95', help='path of probability-based PLA')
    parser.add_argument('--path_soft', type=str, default='Pseudo/pretrain_warmup_soft/LP0.0', help='soft pseudo label for rectification')
    parser.add_argument("--train_thred", default=0, type=float)
    parser.add_argument('--used_save_pseudo', action='store_true', help='if True used saved pseudo label')
    parser.add_argument('--no_droplast', action='store_true')

    parser.add_argument('--resize', type=int, default=2200, help='resize long size')
    parser.add_argument('--rcrop', type=str, default='896,512', help='rondom crop size')
    parser.add_argument('--hflip', type=float, default=0.5, help='random flip probility')

    parser.add_argument('--n_class', type=int, default=19, help='19|16|13')
    parser.add_argument('--num_workers', type=int, default=6)
    #loss
    parser.add_argument('--gan', type=str, default='LS', help='Vanilla|LS')
    parser.add_argument('--adv', type=float, default=0.01, help='loss weight of adv loss, only use when stage=warm_up')
    parser.add_argument('--S_pseudo_src', type=float, default=0.0, help='loss weight of pseudo label for strong augmentation of source')
    parser.add_argument("--rce", action='store_true', help="if true, use symmetry cross entropy loss")
    parser.add_argument("--rce_alpha", default=0.1, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument("--rce_beta", default=1.0, type=float, help="loss weight for symmetry cross entropy loss")
    parser.add_argument("--regular_w", default=0, type=float, help='loss weight for regular term')
    parser.add_argument("--regular_type", default='MRKLD', type=str, help='MRENT|MRKLD')
    parser.add_argument('--proto_consistW', type=float, default=1.0, help='loss weight for proto_consist')
    parser.add_argument("--distillation", default=0, type=float, help="kl loss weight")

    parser.add_argument('--S_pseudo', type=float, default=0.0, help='loss weight of pseudo label for strong augmentation')

    #print
    parser.add_argument('--print_interval', type=int, default=20, help='print loss')
    parser.add_argument('--val_interval', type=int, default=1000, help='validate model iter')

    parser.add_argument('--noshuffle', action='store_true', help='do not use shuffle')
    parser.add_argument('--noaug', action='store_true', help='do not use data augmentation')

    parser.add_argument('--proto_rectify', action='store_true')
    parser.add_argument('--proto_temperature', type=float, default=1.0)
    #stage2
    parser.add_argument("--threshold", default=-1, type=float)
    return parser

def relative_path_to_absolute_path(opt):
    opt.rcrop = [int(opt.rcrop.split(',')[0]), int(opt.rcrop.split(',')[1])]
    opt.resume_path = os.path.join(opt.root, 'Code/ProDA', opt.resume_path)
    opt.src_rootpath = os.path.join(opt.root, opt.src_rootpath)
    opt.tgt_rootpath = os.path.join(opt.root, opt.tgt_rootpath)
    opt.path_LP = os.path.join(opt.root, 'Code/ProDA', opt.path_LP)
    opt.path_soft = os.path.join(opt.root, 'Code/ProDA', opt.path_soft)
    opt.logdir = os.path.join(opt.root, 'Code/ProDA', 'logs', opt.name)
    return opt