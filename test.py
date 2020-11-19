import os 
import torch 
import torch.nn as nn
import numpy as np
from easydict import EasyDict as edict
import logging
import cv2
import time


from network_factory import get_network
from loader_factory import get_loader
from utils import load_test_checkpoints, CalculateAcc, \
    SelfData, load_cfg, print_to_screen


def test(test_loader, model, logger=None, Writer=None):
    
    model.eval()
    with torch.no_grad():
        for its, (img_line, img_noise) in enumerate(test_loader):
            img_line = img_line.cuda() if torch.cuda.is_available() else img_line
            img_noise = img_noise.cuda() if torch.cuda.is_available() else img_noise
            g_results = model(torch.cat((img_line, img_noise), 1))
            for i in range(img_line.shape[0]):
                img_line_test = img_line[i].cpu().numpy().transpose((1,2,0)) * 255
                img_line_test = img_line_test.squeeze()
                cv2.imwrite((cfg.PATH.RES_TEST+"line_{}.jpg".format(i+its)), img_line_test)

                img_res_test = g_results[i].cpu().numpy().transpose((1,2,0)) * 255
                cv2.imwrite((cfg.PATH.RES_TEST+"res_{}.jpg".format(i+its)), img_res_test)
                print("{}/{}".format(i+its,its_num))


def embedding(val_loader,val_loader_1, model_0, model_1, model_2, logger, print_fre=50):
    model_0.eval()
    model_1.eval()
    model_2.eval()
    its_num = len(val_loader)
    acc_single_val = CalculateAcc()
    data_begin = time.time()
    with torch.no_grad():
        for its, (load_0, load_1) in enumerate(zip(val_loader,val_loader_1)):
            data_time = time.time()-data_begin
            imgs, targets, imgs_1 = load_0[0], load_0[1], load_1[0]
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            outputs_0 = model_0(imgs)
            outputs_1 = model_1(imgs_1)
            outputs_2 = model_2(imgs)
            outputs = (outputs_0 + outputs_1 + outputs_2)/3
            loss = 0
            train_time = time.time()-(data_time+data_begin)
            data_begin = time.time()
            acc_single_val.add_value(outputs.cpu(),targets.cpu())
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            if its % print_fre == 0:
                print_to_screen(loss, 0, its, 0, its_num,
                    logger, data_time, train_time, mem, acc=acc_single_val.print_())  
    return acc_single_val.print_()


def val(val_loader, model, logger=None, loss_function=None, epoch=0, print_fre=50):  
    model.eval()
    its_num = len(val_loader)
    acc_single_val = CalculateAcc()
    loss_val = SelfData()
    data_begin = time.time()
    with torch.no_grad():
        for its, (imgs, targets) in enumerate(val_loader):
            data_time = time.time()-data_begin
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            outputs = model(imgs)
            loss = loss_function(outputs,targets) if loss_function is not None else torch.tensor(0)
            train_time = time.time()-(data_time+data_begin)
            data_begin = time.time()
            loss_val.add_value(loss.cpu())
            acc_single_val.add_value(outputs.cpu(),targets.cpu())
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            if its % print_fre == 0:
                print_to_screen(loss, 0, its, epoch, its_num,
                    logger, data_time, train_time, mem, acc=acc_single_val.print_())               
    return acc_single_val.print_(), loss_val.avg()        
        

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print ("Usage: python test.py [eval, test, embedding]")
    else:
        cmd = sys.argv[1]
        from config_cub import cfg
        logger = load_cfg(cfg)
        model = get_network(cfg.MODEL.NAME, cfg=cfg.MODEL, logger=logger)
        model = torch.nn.DataParallel(model, cfg.GPUS).cuda() if torch.cuda.is_available() else model
        load_test_checkpoints(model, cfg.PATH, logger, use_best=True)
        
        if cmd == 'eval':   
            test_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.EVAL, 'eval',label_path=cfg.PATH.LABEL, cfg=cfg.TRAIN, logger=logger)
            acc_val, _ = val(test_loader, model, logger, print_fre=cfg.PRINT_FRE,)
            logger.info("Prec@1:%.4f"%(acc_val))

        elif cmd =='test':
            test_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.TEST, 'test', cfg.TRAIN, logger)
            test(test_loader, model, logger)

        elif cmd =='embedding':
            test_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.EVAL, 'eval',label_path=cfg.PATH.LABEL, cfg=cfg.TRAIN, logger=logger)
            from config_cub_1 import cfg1
            logger = load_cfg(cfg1)
            model_1 = get_network(cfg1.MODEL.NAME, cfg=cfg1.MODEL, logger=logger)
            model_1 = torch.nn.DataParallel(model_1, cfg1.GPUS).cuda() if torch.cuda.is_available() else model_1
            load_test_checkpoints(model_1, cfg1.PATH, logger, use_best=True)
            test_loader_1 = get_loader(cfg1.DATASET_TRPE, cfg1.PATH.EVAL, 'eval',label_path=cfg1.PATH.LABEL, cfg=cfg1.TRAIN, logger=logger)
            from config_cub_2 import cfg2
            logger = load_cfg(cfg2)
            model_2 = get_network(cfg2.MODEL.NAME, cfg=cfg2.MODEL, logger=logger)
            model_2 = torch.nn.DataParallel(model_2, cfg2.GPUS).cuda() if torch.cuda.is_available() else model_2
            load_test_checkpoints(model_2, cfg2.PATH, logger, use_best=True)
            acc_val = embedding(test_loader, test_loader_1, model,model_1,model_2, logger)
            logger.info("Embedding Prec@1:%.4f"%(acc_val))
        else:
            print ("Usage: python test.py [eval, test]")
   
    
