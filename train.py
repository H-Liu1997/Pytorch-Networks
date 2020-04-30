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

from config import cfg, load_cfg
from network_factory import get_network
from opt_factory import get_opt
from loss_factory import get_loss_func
from datasets.loader_factory import get_loader
from test import val
from utils import CalculateAcc


def print_to_screen(loss, lr, its, epoch, its_num, writer, logger):
    writer.add_scalar('loss', loss, epoch*its_num+its)
    logger.info(("[%d]{%d}/{%d}"%(epoch, its, its_num)+
        " loss:%06f"%(loss)+" lr:%06f"%(lr)))
        

def save_checkpoints(save_path, model, opt, epoch):
    states = { 'model_state': model.state_dict(),
               'epoch': epoch + 1,
               'opt_state': opt.state_dict(),}
    torch.save(states, save_path)


def load_checkpoints(model, opt, save_path, logger):
    try:
        states = torch.load(save_path.EXPS+save_path.NAME+save_path.MODEL)
        model.load_state_dict(states['model_state'])
        opt.load_state_dict(states['opt_state'])
        current_epoch = states['epoch']
        logger.info('loading checkpoints success')
    except:
        current_epoch = 0
        logger.info("no checkpoints")
    return current_epoch


def main():
    #from resnet_ori import ResNet18
    logger = load_cfg()     
    torch.manual_seed(cfg.DETERMINISTIC.SEED)
    torch.cuda.manual_seed(cfg.DETERMINISTIC.SEED)
    torch.backends.cudnn.deterministic = cfg.DETERMINISTIC.CUDNN
    np.random.seed(cfg.DETERMINISTIC.SEED)
    
    train_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.DATA, 'train', cfg=cfg.TRAIN, logger=logger)
    val_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.EVAL, 'eval', cfg=cfg.TRAIN, logger=logger)
    
    #model = ResNet18()
    model = get_network(cfg.MODEL.NAME, cfg=cfg.MODEL, logger=logger)
    model = torch.nn.DataParallel(model, cfg.GPUS).cuda() if torch.cuda.is_available() else model
    opt,lr_scheduler = get_opt(model, cfg.TRAIN, logger)
    loss_func = get_loss_func(cfg.MODEL.LOSS, logger=logger)

    current_epoch = load_checkpoints(model, opt, cfg.PATH , logger)
    log_writter = SummaryWriter(cfg.PATH.EXPS+cfg.PATH.NAME)
    its_num = len(train_loader)

    for epoch in range(current_epoch, cfg.TRAIN.EPOCHS):
        acc_train_class = CalculateAcc()
        model.train()
        for its, (imgs, targets)in enumerate(train_loader):
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets
           
            opt.zero_grad()
            outputs = model(imgs)
            loss = loss_func(outputs,targets)
            loss.backward()
            opt.step()

            lr = opt.param_groups[0]['lr']
            acc_train_class.add_value(outputs.cpu(), targets.cpu())
            if its % cfg.PRINT_FRE == 0:
                print_to_screen(loss, lr, its, epoch, its_num, log_writter, logger)
            
            if cfg.SHORT_TEST == True:
                if its == 20:
                    break
        lr_scheduler.step()
        save_checkpoints(cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.MODEL, model, opt, epoch)
        acc_val = val(val_loader, model, logger)
        log_writter.add_scalars("acc",{'acc_train':acc_train_class.print_(),
                                     'acc_val':acc_val,},
                                     epoch)
    log_writter.close()


if __name__ == "__main__":
    main()            