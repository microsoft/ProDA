# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch.nn as nn
import torch.nn.functional as F
import os, sys
import torch
import numpy as np

from models.sync_batchnorm import SynchronizedBatchNorm2d, DataParallelWithCallback
from models.deeplabv2 import Deeplab
from models.discriminator import FCDiscriminator
from .utils import freeze_bn, get_scheduler, cross_entropy2d
from data.randaugment import affine_sample

class feat_prototype_distance_module(nn.Module):
    def __init__(self):
        super(feat_prototype_distance_module, self).__init__()

    def forward(self, feat, objective_vectors, class_numbers):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, class_numbers, H, W)).to(feat.device)
        for i in range(class_numbers):
            #feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
            feat_proto_distance[:, i, :, :] = torch.norm(objective_vectors[0, i].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)
        return feat_proto_distance

class CustomModel():
    def __init__(self, opt, logger, isTrain=True):
        self.opt = opt
        self.class_numbers = opt.n_class
        self.logger = logger
        self.best_iou = -100
        self.nets = []
        self.nets_DP = []
        self.default_gpu = 0
        self.objective_vectors = torch.zeros([self.class_numbers, 256])
        self.objective_vectors_num = torch.zeros([self.class_numbers])

        if opt.bn == 'sync_bn':
            BatchNorm = SynchronizedBatchNorm2d
        elif opt.bn == 'bn':
            BatchNorm = nn.BatchNorm2d
        else:
            raise NotImplementedError('batch norm choice {} is not implemented'.format(opt.bn))

        if self.opt.no_resume:
            restore_from = None
        else:
            restore_from= opt.resume_path
            self.best_iou = 0
        if self.opt.student_init == 'imagenet':
            self.BaseNet = Deeplab(BatchNorm, num_classes=self.class_numbers, freeze_bn=False, restore_from=restore_from)
        elif self.opt.student_init == 'simclr':
            self.BaseNet = Deeplab(BatchNorm, num_classes=self.class_numbers, freeze_bn=False, restore_from=restore_from, 
                initialization=os.path.join(opt.root, 'Code/ProDA', 'pretrained/simclr/r101_1x_sk0.pth'), bn_clr=opt.bn_clr)
        else:
            self.BaseNet = Deeplab(BatchNorm, num_classes=self.class_numbers, freeze_bn=False, restore_from=restore_from)
            
        logger.info('the backbone is {}'.format(opt.model_name))

        self.nets.extend([self.BaseNet])

        self.optimizers = []
        self.schedulers = []        
        optimizer_cls = torch.optim.SGD
        optimizer_params = {'lr':opt.lr, 'weight_decay':2e-4, 'momentum':0.9}

        if self.opt.stage == 'warm_up':
            self.net_D = FCDiscriminator(inplanes=self.class_numbers)
            self.net_D_DP = self.init_device(self.net_D, gpu_id=self.default_gpu, whether_DP=True)
            self.nets.extend([self.net_D])
            self.nets_DP.append(self.net_D_DP)

            self.optimizer_D = torch.optim.Adam(self.net_D.parameters(), lr=1e-4, betas=(0.9, 0.99))
            self.optimizers.extend([self.optimizer_D])
            self.DSchedule = get_scheduler(self.optimizer_D, opt)
            self.schedulers.extend([self.DSchedule])

        if self.opt.finetune or self.opt.stage == 'warm_up':
            self.BaseOpti = optimizer_cls([{'params':self.BaseNet.get_1x_lr_params(), 'lr':optimizer_params['lr']},
                                           {'params':self.BaseNet.get_10x_lr_params(), 'lr':optimizer_params['lr']*10}], **optimizer_params)
        else:
            self.BaseOpti = optimizer_cls(self.BaseNet.parameters(), **optimizer_params)
        self.optimizers.extend([self.BaseOpti])

        self.BaseSchedule = get_scheduler(self.BaseOpti, opt)
        self.schedulers.extend([self.BaseSchedule])

        if self.opt.ema:
            self.BaseNet_ema = Deeplab(BatchNorm, num_classes=self.class_numbers, freeze_bn=False, restore_from=restore_from, bn_clr=opt.ema_bn)
            self.BaseNet_ema.load_state_dict(self.BaseNet.state_dict().copy())

        if self.opt.distillation > 0:
            self.teacher = Deeplab(BatchNorm, num_classes=self.class_numbers, freeze_bn=False, restore_from=opt.resume_path, bn_clr=opt.ema_bn)
            self.teacher.eval()
            self.teacher_DP = self.init_device(self.teacher, gpu_id=self.default_gpu, whether_DP=True)


        self.adv_source_label = 0
        self.adv_target_label = 1
        if self.opt.gan == 'Vanilla':
            self.bceloss = nn.BCEWithLogitsLoss(size_average=True)
        elif self.opt.gan == 'LS':
            self.bceloss = torch.nn.MSELoss()
        self.feat_prototype_distance_DP = self.init_device(feat_prototype_distance_module(), gpu_id=self.default_gpu, whether_DP=True)

        self.BaseNet_DP = self.init_device(self.BaseNet, gpu_id=self.default_gpu, whether_DP=True)
        self.nets_DP.append(self.BaseNet_DP)
        if self.opt.ema:
            self.BaseNet_ema_DP = self.init_device(self.BaseNet_ema, gpu_id=self.default_gpu, whether_DP=True)

    def calculate_mean_vector(self, feat_cls, outputs, labels=None, thresh=None):
        outputs_softmax = F.softmax(outputs, dim=1)
        if thresh is None:
            thresh = -1
        conf = outputs_softmax.max(dim=1, keepdim=True)[0]
        mask = conf.ge(thresh)
        outputs_argmax = outputs_softmax.argmax(dim=1, keepdim=True)
        outputs_argmax = self.process_label(outputs_argmax.float())
        if labels is None:
            outputs_pred = outputs_argmax
        else:
            labels_expanded = self.process_label(labels)
            outputs_pred = labels_expanded * outputs_argmax
        scale_factor = F.adaptive_avg_pool2d(outputs_pred * mask, 1)
        vectors = []
        ids = []
        for n in range(feat_cls.size()[0]):
            for t in range(self.class_numbers):
                if scale_factor[n][t].item()==0:
                    continue
                if (outputs_pred[n][t] > 0).sum() < 10:
                    continue
                s = feat_cls[n] * outputs_pred[n][t] * mask[n]
                # scale = torch.sum(outputs_pred[n][t]) / labels.shape[2] / labels.shape[3] * 2
                # s = normalisation_pooling()(s, scale)
                s = F.adaptive_avg_pool2d(s, 1) / scale_factor[n][t]
                vectors.append(s)
                ids.append(t)
        return vectors, ids

    def step_adv(self, source_x, source_label, target_x, source_imageS, source_params):
        for param in self.net_D.parameters():
            param.requires_grad = False
        self.BaseOpti.zero_grad()
        
        if self.opt.S_pseudo_src > 0:
            source_output = self.BaseNet_DP(source_imageS)
            source_label_d4 = F.interpolate(source_label.unsqueeze(1).float(), size=source_output['out'].size()[2:])
            source_labelS = self.label_strong_T(source_label_d4.clone().float(), source_params, padding=250, scale=4).to(torch.int64)
            loss_ = cross_entropy2d(input=source_output['out'], target=source_labelS.squeeze(1))
            loss_GTA = loss_ * self.opt.S_pseudo_src
            source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)
        else:
            source_output = self.BaseNet_DP(source_x, ssl=True)
            source_outputUp = F.interpolate(source_output['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)

            loss_GTA = cross_entropy2d(input=source_outputUp, target=source_label, size_average=True, reduction='mean')

        target_output = self.BaseNet_DP(target_x, ssl=True)
        target_outputUp = F.interpolate(target_output['out'], size=target_x.size()[2:], mode='bilinear', align_corners=True)
        target_D_out = self.net_D_DP(F.softmax(target_outputUp, dim=1))
        loss_adv_G = self.bceloss(target_D_out, torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_source_label).to(target_D_out.device)) * self.opt.adv
        loss_G = loss_adv_G + loss_GTA
        loss_G.backward()
        self.BaseOpti.step()

        for param in self.net_D.parameters():
            param.requires_grad = True
        self.optimizer_D.zero_grad()
        source_D_out = self.net_D_DP(F.softmax(source_outputUp.detach(), dim=1))
        target_D_out = self.net_D_DP(F.softmax(target_outputUp.detach(), dim=1))
        loss_D = self.bceloss(source_D_out, torch.FloatTensor(source_D_out.data.size()).fill_(self.adv_source_label).to(source_D_out.device)) + \
                    self.bceloss(target_D_out, torch.FloatTensor(target_D_out.data.size()).fill_(self.adv_target_label).to(target_D_out.device))
        loss_D.backward()
        self.optimizer_D.step()

        return loss_GTA.item(), loss_adv_G.item(), loss_D.item()

    def step(self, source_x, source_label, target_x, target_imageS=None, target_params=None, target_lp=None, 
            target_lpsoft=None, target_image_full=None, target_weak_params=None):

        source_out = self.BaseNet_DP(source_x, ssl=True)
        source_outputUp = F.interpolate(source_out['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)

        loss_GTA = cross_entropy2d(input=source_outputUp, target=source_label)
        loss_GTA.backward()        

        if self.opt.proto_rectify:
            threshold_arg = F.interpolate(target_lpsoft, scale_factor=0.25, mode='bilinear', align_corners=True)
        else:
            threshold_arg = F.interpolate(target_lp.unsqueeze(1).float(), scale_factor=0.25).long()

        if self.opt.ema:
            ema_input = target_image_full
            with torch.no_grad():
                ema_out = self.BaseNet_ema_DP(ema_input)
            ema_out['feat'] = F.interpolate(ema_out['feat'], size=(int(ema_input.shape[2]/4), int(ema_input.shape[3]/4)), mode='bilinear', align_corners=True)
            ema_out['out'] = F.interpolate(ema_out['out'], size=(int(ema_input.shape[2]/4), int(ema_input.shape[3]/4)), mode='bilinear', align_corners=True)

        target_out = self.BaseNet_DP(target_imageS) if self.opt.S_pseudo > 0 else self.BaseNet_DP(target_x)
        target_out['out'] = F.interpolate(target_out['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
        target_out['feat'] = F.interpolate(target_out['feat'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)

        loss = torch.Tensor([0]).to(self.default_gpu)
        batch, _, w, h = threshold_arg.shape
        if self.opt.proto_rectify:
            weights = self.get_prototype_weight(ema_out['feat'], target_weak_params=target_weak_params)
            rectified = weights * threshold_arg
            threshold_arg = rectified.max(1, keepdim=True)[1]
            rectified = rectified / rectified.sum(1, keepdim=True)
            argmax = rectified.max(1, keepdim=True)[0]
            threshold_arg[argmax < self.opt.train_thred] = 250
        if self.opt.S_pseudo > 0:
            threshold_argS = self.label_strong_T(threshold_arg.clone().float(), target_params, padding=250, scale=4).to(torch.int64)
            cluster_argS = self.label_strong_T(cluster_arg.clone().float(), target_params, padding=250, scale=4).to(torch.int64)
            threshold_arg = threshold_argS

        loss_CTS = cross_entropy2d(input=target_out['out'], target=threshold_arg.reshape([batch, w, h]))

        if self.opt.rce:
            rce = self.rce(target_out['out'], threshold_arg.reshape([batch, w, h]).clone())
            loss_CTS = self.opt.rce_alpha * loss_CTS + self.opt.rce_beta * rce

        if self.opt.regular_w > 0:
            regular_loss = self.regular_loss(target_out['out'])
            loss_CTS = loss_CTS + regular_loss * self.opt.regular_w

        cluster_argS = None
        loss_consist = torch.Tensor([0]).to(self.default_gpu)
        if self.opt.proto_consistW > 0:
            ema2weak_feat = self.full2weak(ema_out['feat'], target_weak_params)         #N*256*H*W
            ema2weak_feat_proto_distance = self.feat_prototype_distance(ema2weak_feat)  #N*19*H*W
            ema2strong_feat_proto_distance = self.label_strong_T(ema2weak_feat_proto_distance, target_params, padding=250, scale=4)
            mask = (ema2strong_feat_proto_distance != 250).float()
            teacher = F.softmax(-ema2strong_feat_proto_distance * self.opt.proto_temperature, dim=1)

            targetS_out = target_out if self.opt.S_pseudo > 0 else self.BaseNet_DP(target_imageS)
            targetS_out['out'] = F.interpolate(targetS_out['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
            targetS_out['feat'] = F.interpolate(targetS_out['feat'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)

            prototype_tmp = self.objective_vectors.expand(4, -1, -1)  #gpu memory limitation
            strong_feat_proto_distance = self.feat_prototype_distance_DP(targetS_out['feat'], prototype_tmp, self.class_numbers)
            student = F.log_softmax(-strong_feat_proto_distance * self.opt.proto_temperature, dim=1)

            loss_consist = F.kl_div(student, teacher, reduction='none')
            loss_consist = (loss_consist * mask).sum() / mask.sum()
            loss = loss + self.opt.proto_consistW * loss_consist

        loss = loss + loss_CTS
        loss.backward()
        self.BaseOpti.step()
        self.BaseOpti.zero_grad()

        if self.opt.moving_prototype: #update prototype
            ema_vectors, ema_ids = self.calculate_mean_vector(ema_out['feat'].detach(), ema_out['out'].detach())
            for t in range(len(ema_ids)):
                self.update_objective_SingleVector(ema_ids[t], ema_vectors[t].detach(), start_mean=False)
        
        if self.opt.ema: #update ema model
            for param_q, param_k in zip(self.BaseNet.parameters(), self.BaseNet_ema.parameters()):
                param_k.data = param_k.data.clone() * 0.999 + param_q.data.clone() * (1. - 0.999)
            for buffer_q, buffer_k in zip(self.BaseNet.buffers(), self.BaseNet_ema.buffers()):
                buffer_k.data = buffer_q.data.clone()

        return loss.item(), loss_CTS.item(), loss_consist.item()

    def regular_loss(self, activation):
        logp = F.log_softmax(activation, dim=1)
        if self.opt.regular_type == 'MRENT':
            p = F.softmax(activation, dim=1)
            loss = (p * logp).sum() / (p.shape[0]*p.shape[2]*p.shape[3])
        elif self.opt.regular_type == 'MRKLD':
            loss = - logp.sum() / (logp.shape[0]*logp.shape[1]*logp.shape[2]*logp.shape[3])
        return loss

    def rce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        mask = (labels != 250).float()
        labels[labels==250] = self.class_numbers
        label_one_hot = torch.nn.functional.one_hot(labels, self.class_numbers + 1).float().to(self.default_gpu)
        label_one_hot = torch.clamp(label_one_hot.permute(0,3,1,2)[:,:-1,:,:], min=1e-4, max=1.0)
        rce = -(torch.sum(pred * torch.log(label_one_hot), dim=1) * mask).sum() / (mask.sum() + 1e-6)
        return rce

    def step_distillation(self, source_x, source_label, target_x, target_imageS=None, target_params=None, target_lp=None):

        source_out = self.BaseNet_DP(source_x, ssl=True)
        source_outputUp = F.interpolate(source_out['out'], size=source_x.size()[2:], mode='bilinear', align_corners=True)
        loss_GTA = cross_entropy2d(input=source_outputUp, target=source_label)
        loss_GTA.backward()

        threshold_arg = F.interpolate(target_lp.unsqueeze(1).float(), scale_factor=0.25).long()
        if self.opt.S_pseudo > 0:
            threshold_arg = self.label_strong_T(threshold_arg.clone().float(), target_params, padding=250, scale=4).to(torch.int64)
            target_out = self.BaseNet_DP(target_imageS)
        else:
            target_out = self.BaseNet_DP(target_x)
        target_out['out'] = F.interpolate(target_out['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
        batch, _, w, h = threshold_arg.shape
        loss = cross_entropy2d(input=target_out['out'], target=threshold_arg.reshape([batch, w, h]), size_average=True, reduction='mean')
        if self.opt.rce:
            rce = self.rce(target_out['out'], threshold_arg.reshape([batch, w, h]).clone())
            loss = self.opt.rce_alpha * loss + self.opt.rce_beta * rce

        if self.opt.distillation > 0:
            student = F.softmax(target_out['out'], dim=1)
            with torch.no_grad():
                teacher_out = self.teacher_DP(target_imageS)
                teacher_out['out'] = F.interpolate(teacher_out['out'], size=threshold_arg.shape[2:], mode='bilinear', align_corners=True)
                teacher = F.softmax(teacher_out['out'], dim=1)

            loss_kd = F.kl_div(student, teacher, reduction='none')
            mask = (teacher != 250).float()
            loss_kd = (loss_kd * mask).sum() / mask.sum()
            loss = loss + self.opt.distillation * loss_kd

        loss.backward()
        self.BaseOpti.step()
        self.BaseOpti.zero_grad()
        return loss_GTA.item(), loss.item()

    def full2weak(self, feat, target_weak_params):
        tmp = []
        for i in range(feat.shape[0]):
            h, w = target_weak_params['RandomSized'][0][i], target_weak_params['RandomSized'][1][i]
            feat_ = F.interpolate(feat[i:i+1], size=[int(h/4), int(w/4)], mode='bilinear', align_corners=True)
            y1, y2, x1, x2 = target_weak_params['RandomCrop'][0][i], target_weak_params['RandomCrop'][1][i], target_weak_params['RandomCrop'][2][i], target_weak_params['RandomCrop'][3][i]
            y1, th, x1, tw = int(y1/4), int((y2-y1)/4), int(x1/4), int((x2-x1)/4)
            feat_ = feat_[:, :, y1:y1+th, x1:x1+tw]
            if target_weak_params['RandomHorizontallyFlip'][i]:
                inv_idx = torch.arange(feat_.size(3)-1,-1,-1).long().to(feat_.device)
                feat_ = feat_.index_select(3,inv_idx)
            tmp.append(feat_)
        feat = torch.cat(tmp, 0)
        return feat

    def feat_prototype_distance(self, feat):
        N, C, H, W = feat.shape
        feat_proto_distance = -torch.ones((N, self.class_numbers, H, W)).to(feat.device)
        for i in range(self.class_numbers):
            #feat_proto_distance[:, i, :, :] = torch.norm(torch.Tensor(self.objective_vectors[i]).reshape(-1,1,1).expand(-1, H, W).to(feat.device) - feat, 2, dim=1,)
            feat_proto_distance[:, i, :, :] = torch.norm(self.objective_vectors[i].reshape(-1,1,1).expand(-1, H, W) - feat, 2, dim=1,)
        return feat_proto_distance

    def get_prototype_weight(self, feat, label=None, target_weak_params=None):
        feat = self.full2weak(feat, target_weak_params)
        feat_proto_distance = self.feat_prototype_distance(feat)
        feat_nearest_proto_distance, feat_nearest_proto = feat_proto_distance.min(dim=1, keepdim=True)

        feat_proto_distance = feat_proto_distance - feat_nearest_proto_distance
        weight = F.softmax(-feat_proto_distance * self.opt.proto_temperature, dim=1)
        return weight

    def label_strong_T(self, label, params, padding, scale=1):
        label = label + 1
        for i in range(label.shape[0]):
            for (Tform, param) in params.items():
                if Tform == 'Hflip' and param[i].item() == 1:
                    label[i] = label[i].clone().flip(-1)
                elif (Tform == 'ShearX' or Tform == 'ShearY' or Tform == 'TranslateX' or Tform == 'TranslateY' or Tform == 'Rotate') and param[i].item() != 1e4:
                    v = int(param[i].item() // scale) if Tform == 'TranslateX' or Tform == 'TranslateY' else param[i].item()
                    label[i:i+1] = affine_sample(label[i:i+1].clone(), v, Tform)
                elif Tform == 'CutoutAbs' and isinstance(param, list):
                    x0 = int(param[0][i].item() // scale)
                    y0 = int(param[1][i].item() // scale)
                    x1 = int(param[2][i].item() // scale)
                    y1 = int(param[3][i].item() // scale)
                    label[i, :, y0:y1, x0:x1] = 0
        label[label == 0] = padding + 1  # for strong augmentation, constant padding
        label = label - 1
        return label

    def process_label(self, label):
        batch, channel, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).to(self.default_gpu)
        id = torch.where(label < self.class_numbers, label, torch.Tensor([self.class_numbers]).to(self.default_gpu))
        pred1 = pred1.scatter_(1, id.long(), 1)
        return pred1

    def freeze_bn_apply(self):
        for net in self.nets:
            net.apply(freeze_bn)
        for net in self.nets_DP:
            net.apply(freeze_bn)

    def scheduler_step(self):
        for scheduler in self.schedulers:
            scheduler.step()
    
    def optimizer_zerograd(self):
        for optimizer in self.optimizers:
            optimizer.zero_grad()
    

    def init_device(self, net, gpu_id=None, whether_DP=False):
        gpu_id = gpu_id or self.default_gpu
        device = torch.device("cuda:{}".format(gpu_id) if torch.cuda.is_available() else 'cpu')
        net = net.to(device)
        # if torch.cuda.is_available():
        if whether_DP:
            #net = DataParallelWithCallback(net, device_ids=[0])
            net = DataParallelWithCallback(net, device_ids=range(torch.cuda.device_count()))
        return net
    
    def eval(self, net=None, logger=None):
        """Make specific models eval mode during test time"""
        # if issubclass(net, nn.Module) or issubclass(net, BaseModel):
        if net == None:
            for net in self.nets:
                net.eval()
            for net in self.nets_DP:
                net.eval()
            if logger!=None:    
                logger.info("Successfully set the model eval mode") 
        else:
            net.eval()
            if logger!=None:    
                logger("Successfully set {} eval mode".format(net.__class__.__name__))
        return

    def train(self, net=None, logger=None):
        if net==None:
            for net in self.nets:
                net.train()
            for net in self.nets_DP:
                net.train()
        else:
            net.train()
        return

    def update_objective_SingleVector(self, id, vector, name='moving_average', start_mean=True):
        if vector.sum().item() == 0:
            return
        if start_mean and self.objective_vectors_num[id].item() < 100:
            name = 'mean'
        if name == 'moving_average':
            self.objective_vectors[id] = self.objective_vectors[id] * (1 - self.opt.proto_momentum) + self.opt.proto_momentum * vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
        elif name == 'mean':
            self.objective_vectors[id] = self.objective_vectors[id] * self.objective_vectors_num[id] + vector.squeeze()
            self.objective_vectors_num[id] += 1
            self.objective_vectors[id] = self.objective_vectors[id] / self.objective_vectors_num[id]
            self.objective_vectors_num[id] = min(self.objective_vectors_num[id], 3000)
            pass
        else:
            raise NotImplementedError('no such updating way of objective vectors {}'.format(name))

