import os 
import torch 
import torch.nn as nn
import numpy as np
from easydict import EasyDict as edict
import logging
import cv2
import time


from network_factory import get_network
from datasets.loader_factory import get_loader
from config import cfg
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


def val(val_loader, model, logger=None, loss_function=None, epoch=0, print_fre=50):  
    its_num = len(val_loader)
    model.eval()
    acc_single_val = CalculateAcc()
    loss_val = SelfData()
    data_begin = time.time()
    with torch.no_grad():
        for its, (imgs, targets) in enumerate(val_loader):
            data_time = time.time()-data_begin
            imgs = imgs.cuda() if torch.cuda.is_available() else imgs
            targets = targets.cuda() if torch.cuda.is_available() else targets
            outputs = model(imgs)
            loss = loss_function(outputs,targets)
            train_time = time.time()-(data_time+data_begin)
            data_begin = time.time()
            loss_val.add_value(loss.cpu())
            acc_single_val.add_value(outputs.cpu(),targets.cpu())
            mem = torch.cuda.memory_cached() / 1E9 if torch.cuda.is_available() else 0
            if its % print_fre == 0:
                print_to_screen(loss, 0, its, epoch, its_num,
                    logger, data_time, train_time, mem, acc=acc_single_val.print_())               
    return acc_single_val.print_(), loss_val.avg()        
    

def main(test_type):
    logger = load_cfg(cfg)
    model = get_network(cfg.MODEL.NAME, logger=logger)
    model = torch.nn.DataParallel(model, cfg.GPUS).cuda() if torch.cuda.is_available() else model
    load_test_checkpoints(model, cfg.PATH, logger)
    if test_type == 'eval':    
        test_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.EVAL, 'eval', cfg.TRAIN, logger)
        acc = val(test_loader, model, logger)
    else:
        test_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.TEST, 'test', cfg.TRAIN, logger)
        test(test_loader, model, logger)
    

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print ("Usage: python test.py [eval, test]")
    else:
        cmd = sys.argv[1]
        if cmd == "eval":
            main("eval")
        elif cmd == "test":
            main("test")  
        else:
            print ("Usage: python test.py [eval, test]")
   
    
