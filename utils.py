import os
from collections import OrderedDict
import torch
import logging
from easydict import EasyDict as edict
import yaml


def print_to_screen(loss, lr, its, epoch, its_num,
    logger, data_time, train_time, mem, acc=0):
    logger.info(("[%d][%d/%d]\t"%(epoch, its, its_num)+
        "Loss:%.5f\t"%(loss)+"Lr:%.6f\t"%(lr)
        +"Data:%.4dms\t"%(data_time*1000)+"Train:%.4dms\t"%(train_time*1000))
        +"Mem:%.2fGb\t"%(mem) +"Prec@1:%.4f"%(acc))
        

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


def model_complexity(model,cfg,logger):
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model,  (cfg.MODEL.IN_DIM, cfg.TRAIN.CROP, cfg.TRAIN.CROP), 
        as_strings=True, print_per_layer_stat=True)
    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))


def load_cfg(cfg):
    cfg_name = cfg.PATH.EXPS+cfg.PATH.NAME+'/'+cfg.PATH.NAME+'.yaml'
    if not os.path.exists(cfg.PATH.EXPS+cfg.PATH.NAME):
            os.mkdir(cfg.PATH.EXPS+cfg.PATH.NAME)
    # for log path, can only change by code file
    logging.basicConfig(format='%(message)s',
        level=logging.DEBUG,
        filename=cfg.PATH.EXPS+cfg.PATH.NAME+cfg.PATH.LOG)
    stream_handler = logging.StreamHandler()
    logger = logging.getLogger(cfg.PATH.NAME)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(stream_handler)

    if os.path.exists(cfg_name):
        logger.info('start loading config files...')
        seed_add = 10 
        with open(cfg_name) as f:
            old_cfg = edict(yaml.load(f, Loader=yaml.FullLoader))
            for k, v in old_cfg.items():
                if k in cfg:
                    if isinstance(v, dict):
                        for vk, vv in v.items():
                            if vk in cfg[k]:
                                cfg[k][vk] = vv
                            else:
                                logger.error("{} not exist in config.py".format(vk))
                    else:
                        cfg[k] = v   
                else:
                   logger.error("{} not exist in config.py".format(k))
        logger.info('loading config files success')
        cfg.DETERMINISTIC.SEED += seed_add
        logger.info('change random seed success')
    else:
        logger.info('start creating config files...')
    cfg_dict = dict(cfg)
    for k, v in cfg_dict.items():
        if isinstance(v, edict):
            cfg_dict[k] = dict(v)
    with open(cfg_name, 'w') as f:
        yaml.dump(dict(cfg_dict), f, default_flow_style=False)
    logger.info('update config files success')
    return logger

    
class SelfData(object):
    def __init__(self):
        self.value = 0
        self.counter = 0 + 1e-8

    def add_value(self,add_value):
        self.counter += 1
        self.value += add_value.data.cpu().numpy()

    def avg(self):
        return self.value/self.counter


class CalculateAcc(object):
    def __init__(self,topk=1):
        self.count_success_a = 0 + 1e-8
        self.count = 0+ 1e-8
        self.topk = topk
        
    def add_value(self,output,target):
        self.count += output.shape[0]
        _, preds = output.data.topk(self.topk,1,True,True)
        preds = preds.t()
        for pred in preds:
            self.count_success_a += pred.eq(target.data.view_as(pred)).sum().numpy()

    def print_(self):
        return (self.count_success_a/self.count)


def load_test_checkpoints(model, save_path, logger):
    try:
        #logger.debug(save_path.EXPS+save_path.NAME+save_path.GMODEL)
        states= torch.load(save_path.EXPS+save_path.NAME+save_path.MODEL) if torch.cuda.is_available() \
            else torch.load(save_path.EXPS+save_path.NAME+save_path.MODEL, map_location=torch.device('cpu'))
        #logger.debug("success")
        try:
            model.load_state_dict(states['model_state'])
        except:
            states_no_module = OrderedDict()
            for k, v in states['model_state'].items():
                name_no_module = k[7:]
                states_no_module[name_no_module] = v
            model.load_state_dict(states_no_module)
        logger.info('loading checkpoints success')
    except:
        logger.error("no checkpoints")


def plot_result_data(acc_total, acc_val_total, loss_total, losss_val_total, cfg_path, epoch):
    import matplotlib.pyplot as plt
    y = range(epoch)
    plt.plot(y,acc_total,linestyle="-",  linewidth=1,label='acc_train')
    plt.plot(y,acc_val_total,linestyle="-", linewidth=1,label='acc_val')
    plt.legend(('acc_train', 'acc_val'), loc='upper right')
    plt.xlabel("Training Epoch")
    plt.ylabel("Acc on dataset")
    plt.savefig('{}/acc.png'.format(cfg_path))
    plt.cla()
    plt.plot(y,loss_total,linestyle="-", linewidth=1,label='loss_train')
    plt.plot(y,losss_val_total,linestyle="-", linewidth=1,label='loss_val')
    plt.legend(('loss_train', 'loss_val'), loc='upper right')
    plt.xlabel("Training Epoch")
    plt.ylabel("Loss on dataset")
    plt.savefig('{}/loss.png'.format(cfg_path))
