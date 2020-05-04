import os
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
cfg.PATH.NAME = 'res_110_82_123'
cfg.PATH.MODEL = '/model.pth'
cfg.PATH.BESTMODEL = '/bestmodel.pth'
cfg.PATH.LOG = '/log.txt'
cfg.PATH.RESULTS = '/results/'

cfg.DETERMINISTIC = edict()
cfg.DETERMINISTIC.SEED = 100
cfg.DETERMINISTIC.CUDNN = True

cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 164
cfg.TRAIN.BATCHSIZE = 128
cfg.TRAIN.L1SCALING = 100
cfg.TRAIN.TYPE = 'sgd'
cfg.TRAIN.LR = 1e-1
cfg.TRAIN.BETA1 = 0.9
cfg.TRAIN.BETA2 = 0.999
cfg.TRAIN.LR_TYPR = 'stone'
cfg.TRAIN.LR_REDUCE = [82,123]
cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.NUM_WORKERS = 16
cfg.TRAIN.USE_AUG = True
cfg.TRAIN.CROP = 32
cfg.TRAIN.PAD = 4


cfg.MODEL = edict()
cfg.MODEL.NAME = 'resnet'
cfg.MODEL.BLOCK = 'basicblock'
cfg.MODEL.BLOCK_LIST = [18,18,18,0]
cfg.MODEL.IN_DIM = 3
cfg.MODEL.CLASS_NUM = 10 
cfg.MODEL.BASE = 16 
cfg.MODEL.USE_FC = True 
cfg.MODEL.CONV1 = (3,1,1)
cfg.MODEL.OPERATION = 'A'
cfg.MODEL.STRIDE1 = 1
cfg.MODEL.MAX_POOL = False 
cfg.MODEL.PRETRAINED = False  

cfg.MODEL.LOSS = 'bce_only_g' 

cfg.GPUS = [0]
cfg.PRINT_FRE = 150
cfg.DATASET_TRPE = 'cifar'
cfg.SHORT_TEST = False


if __name__ == "__main__":
    from utils import load_cfg
    logger = load_cfg(cfg)
    print(cfg)




