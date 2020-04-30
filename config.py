import os
import yaml
import logging
from easydict import EasyDict as edict


cfg = edict()
cfg.PATH = edict()
cfg.PATH.DATA = ['/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_1',
                 '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_2',
                 '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_3',
                 '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_4',
                 '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_5',]
cfg.PATH.EVAL = ['/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/test_batch']
cfg.PATH.TEST = '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_1'
cfg.PATH.RES_TEST = './res_imgs/'
cfg.PATH.EXPS = './exps/'
cfg.PATH.NAME = 'res_34_la'
cfg.PATH.MODEL = '/model.pth'
cfg.PATH.LOG = '/log.txt'
cfg.PATH.RESULTS = '/results/'

cfg.DETERMINISTIC = edict()
cfg.DETERMINISTIC.SEED = 0
cfg.DETERMINISTIC.CUDNN = True

cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 200
cfg.TRAIN.BATCHSIZE = 128
cfg.TRAIN.L1SCALING = 100
cfg.TRAIN.TYPE = 'sgd'
cfg.TRAIN.LR = 1e-1
cfg.TRAIN.BETA1 = 0.9
cfg.TRAIN.BETA2 = 0.999
cfg.TRAIN.LR_TYPR = 'stone'
cfg.TRAIN.LR_REDUCE = [140,170]
cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.NUM_WORKERS = 16
cfg.TRAIN.USE_AUG = True
cfg.TRAIN.CROP = 32
cfg.TRAIN.PAD = 4


cfg.MODEL = edict()
cfg.MODEL.NAME = 'resnet'
cfg.MODEL.BLOCK = 'basicblock'
cfg.MODEL.BLOCK_LIST = [5,5,5,0]
cfg.MODEL.IN_DIM = 3
cfg.MODEL.CLASS_NUM = 10 
cfg.MODEL.BASE = 16 
cfg.MODEL.USE_FC = True 
cfg.MODEL.CONV1 = (3,1,1)
cfg.MODEL.MAX_POOL = False 
cfg.MODEL.PRETRAINED = False  

cfg.MODEL.LOSS = 'bce_only_g' 


cfg.GPUS = [0]
cfg.PRINT_FRE = 50
cfg.DATASET_TRPE = 'cifar'
cfg.SHORT_TEST = False



def load_cfg():
    cfg_name = cfg.PATH.EXPS+cfg.PATH.NAME+'/'+cfg.PATH.NAME+'.yaml'
    if not os.path.exists(cfg.PATH.EXPS+cfg.PATH.NAME):
            os.mkdir(cfg.PATH.EXPS+cfg.PATH.NAME)
    # for log path, can only change by code file
    logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s',
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

if __name__ == "__main__":
    logger = load_cfg()
    print(cfg)




