#!usr/bin/env python3
#-*- coding=utf-8 -*-
#python=3.6 pytorch=1.2.0


import os
import random
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from easydict import EasyDict as edict
from random import randint
import torchvision
import torchvision.transforms as transforms

import time

from network_factory import get_network
from opt_factory import get_opt
from loss_factory import get_loss_func
from datasets.loader_factory import get_loader
from test import val
from utils import CalculateAcc,SelfData,load_cfg,model_complexity,plot_result_data,\
    load_checkpoints,print_to_screen,save_checkpoints


def fix_random_seed(cfg):
    random.seed(cfg.DETERMINISTIC.SEED)
    torch.manual_seed(cfg.DETERMINISTIC.SEED)
    torch.cuda.manual_seed(cfg.DETERMINISTIC.SEED)
    torch.backends.cudnn.deterministic = cfg.DETERMINISTIC.CUDNN
    np.random.seed(cfg.DETERMINISTIC.SEED)


def trainer(cfg):
    logger = load_cfg(cfg) 
    fix_random_seed(cfg)    
   
    train_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.DATA, 'train', label_path=cfg.PATH.LABEL, cfg=cfg.TRAIN, logger=logger)
    val_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.EVAL, 'eval',label_path=cfg.PATH.LABEL, cfg=cfg.TRAIN, logger=logger)
    its_num = len(train_loader)
    from backbone.RegNet2020 import RegNet
    # from torchvision import models
    # import torch.nn as nn
    # class ResNet(nn.Module):
    #     def __init__(self, pre_trained=True, n_class=200, model_choice=50):
    #         super(ResNet, self).__init__()
    #         self.n_class = n_class
    #         self.base_model = self._model_choice(pre_trained, model_choice)
    #         self.base_model.avgpool = nn.AdaptiveAvgPool2d((1,1))
    #         self.base_model.fc = nn.Linear(512*4, n_class)
    #         #self.base_model.fc.apply(weight_init_kaiming)

    #     def forward(self, x):
    #         N = x.size(0)
    #         assert x.size() == (N, 3, 224, 224)
    #         x = self.base_model(x)
    #         assert x.size() == (N, self.n_class)
    #         return x

    #     def _model_choice(self, pre_trained, model_choice):
    #         if model_choice == 50:
    #             return models.resnet50(pretrained=pre_trained)
    #         elif model_choice == 101:
    #             return models.resnet101(pretrained=pre_trained)
    #         elif model_choice == 152:
    #             return models.resnet152(pretrained=pre_trained)

    model = RegNet()
    checkpoints = torch.load('./RegNetX-4.0GF_dds_8gpu.pyth')
    from collections import OrderedDict
    states_no_module = OrderedDict()
    for k, v in checkpoints['model_state'].items():
        if k != 'head.fc.weight' and k!= 'head.fc.bias':
            name_no_module = k
            states_no_module[name_no_module] = v
    model.load_state_dict(states_no_module,strict=False)
    #model.load_state_dict(state,strict=False)
    #model = get_network(cfg.MODEL.NAME, cfg=cfg.MODEL, logger=logger)
    model = torch.nn.DataParallel(model, cfg.GPUS).cuda() if torch.cuda.is_available() else model
    model_complexity(model,cfg,logger)

    opt_t,lr_scheduler_t = get_opt(model, cfg.TRAIN, logger, 
        its_total=(cfg.TRAIN.EPOCHS-cfg.TRAIN.WARMUP)*its_num)
    if cfg.TRAIN.WARMUP != 0:
        warm_opt, warm_scheduler = get_opt(model, cfg.TRAIN, logger, 
            is_warm=True, its_total=cfg.TRAIN.WARMUP*its_num)
    loss_func = get_loss_func(cfg.MODEL.LOSS, logger=logger)

    current_epoch = load_checkpoints(model, opt_t, cfg.PATH , logger, lr_scheduler_t)
    log_writter = SummaryWriter(cfg.PATH.EXPS+cfg.PATH.NAME)
    
    #TODO(liu):add warm up
    acc_total = []
    acc_val_total = []
    loss_total = []
    losss_val_total = []
    best_val = [0,0]
    for epoch in range(current_epoch, cfg.TRAIN.EPOCHS):
        if epoch < cfg.TRAIN.WARMUP:
            opt = warm_opt
            lr_scheduler = warm_scheduler
        else:
            opt = opt_t
            lr_scheduler = lr_scheduler_t

        acc_train_class = CalculateAcc()
        loss_train_calss = SelfData()
        model.train()
        data_begin = time.time()
        for its, (imgs, targets)in enumerate(train_loader):
            data_time = time.time()-data_begin
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets
           
            opt.zero_grad()
            outputs = model(imgs)
            loss = loss_func(outputs,targets)
            loss.backward()
            opt.step()
            lr_scheduler.step()
            
            loss_train_calss.add_value(loss.cpu())
            train_time = time.time()-(data_time+data_begin)
            data_begin = time.time()
            lr = opt.param_groups[0]['lr']
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            acc_train_class.add_value(outputs.cpu(), targets.cpu())
            if its % cfg.PRINT_FRE == 0:
                print_to_screen(loss, lr, its, epoch, its_num, logger, 
                    data_time,train_time,mem,acc_train_class.print_())
            
            if cfg.SHORT_TEST == True:
                if its == 20:
                    break

        save_checkpoints(cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.MODEL, model, opt, epoch,lr_scheduler)
        acc_val, loss_val = val(val_loader, model, logger, loss_func, epoch, print_fre=cfg.PRINT_FRE)
        log_writter.add_scalars("acc",{'acc_train':acc_train_class.print_(),
                                     'acc_val':acc_val,},
                                     epoch)
        acc_total.append(acc_train_class.print_())
        acc_val_total.append(acc_val)
        loss_total.append(loss_train_calss.avg())
        losss_val_total.append(loss_val)
        logger.info('Train Prec@1:%.4f\t'%(acc_train_class.print_())+'Val Prec@1:%.4f'%(acc_val))
        if best_val[0] < acc_val:
            best_val[0] = acc_val
            best_val[1] = epoch
            save_checkpoints(cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.BESTMODEL, model, opt, epoch,lr_scheduler)
        logger.info('BestV Prec@1:%.4f\t'%(best_val[0])+"Best Epoch:%d"%(best_val[1]))

    plot_result_data(acc_total,acc_val_total,loss_total,
        losss_val_total,cfg.PATH.EXPS+cfg.PATH.NAME, cfg.TRAIN.EPOCHS)
    log_writter.close()


if __name__ == "__main__":
    from config_cub import cfg
    trainer(cfg)
   
    # from config2 import cfg
    # main(cfg)            