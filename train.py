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
   
    train_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.DATA, 'train', cfg=cfg.TRAIN, logger=logger)
    val_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.EVAL, 'eval', cfg=cfg.TRAIN, logger=logger)
    
    model = get_network(cfg.MODEL.NAME, cfg=cfg.MODEL, logger=logger)
    model = torch.nn.DataParallel(model, cfg.GPUS).cuda() if torch.cuda.is_available() else model
    model_complexity(model,cfg,logger)

    opt,lr_scheduler = get_opt(model, cfg.TRAIN, logger)
    loss_func = get_loss_func(cfg.MODEL.LOSS, logger=logger)

    current_epoch = load_checkpoints(model, opt, cfg.PATH , logger)
    log_writter = SummaryWriter(cfg.PATH.EXPS+cfg.PATH.NAME)
    
    #TODO(liu):add warm up
    acc_total = []
    acc_val_total = []
    loss_total = []
    losss_val_total = []
    its_num = len(train_loader)
    best_val = [0,0]
    for epoch in range(current_epoch, cfg.TRAIN.EPOCHS):
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

        lr_scheduler.step()
        save_checkpoints(cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.MODEL, model, opt, epoch)
        acc_val, loss_val = val(val_loader, model, logger, loss_func, epoch)
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
            save_checkpoints(cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.BESTMODEL, model, opt, epoch)
        logger.info('BestV Prec@1:%.4f\t'%(best_val[0])+"Best Epoch:%d"%(best_val[1]))

    plot_result_data(acc_total,acc_val_total,loss_total,
        losss_val_total,cfg.PATH.EXPS+cfg.PATH.NAME, cfg.TRAIN.EPOCHS)
    log_writter.close()


if __name__ == "__main__":
    from config import cfg
    trainer(cfg)
    # from config2 import cfg
    # main(cfg)            