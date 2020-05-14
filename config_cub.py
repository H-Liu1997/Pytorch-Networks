import os
from easydict import EasyDict as edict


cfg = edict()
cfg.PATH = edict()
#cfg.PATH.DATA = '/home/ikenaga/Public/'
cfg.PATH.DATA =  ['/home/ikenaga/Public/CUB_200_2011/images.txt',
       '/home/ikenaga/Public/CUB_200_2011/train_test_split.txt',
       '/home/ikenaga/Public/CUB_200_2011/images/']
cfg.PATH.LABEL = '/home/ikenaga/Public/CUB_200_2011/image_class_labels.txt'
#cfg.PATH.EVAL = '/home/ikenaga/Public/'
cfg.PATH.EVAL = ['/home/ikenaga/Public/CUB_200_2011/images.txt',
       '/home/ikenaga/Public/CUB_200_2011/train_test_split.txt',
       '/home/ikenaga/Public/CUB_200_2011/images/']
cfg.PATH.TEST = '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_1'
cfg.PATH.RES_TEST = './res_imgs/'
cfg.PATH.EXPS = './exps/'
cfg.PATH.NAME = 'res_50_cub_v12'
cfg.PATH.MODEL = '/model.pth'
cfg.PATH.BESTMODEL = '/bestmodel.pth'
cfg.PATH.LOG = '/log.txt'
cfg.PATH.RESULTS = '/results/'

cfg.DETERMINISTIC = edict()
cfg.DETERMINISTIC.SEED = 60
cfg.DETERMINISTIC.CUDNN = True

cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 120
cfg.TRAIN.BATCHSIZE = 16
cfg.TRAIN.L1SCALING = 100
cfg.TRAIN.TYPE = 'sgd'
cfg.TRAIN.LR = 1e-3
cfg.TRAIN.BETA1 = 0.9
cfg.TRAIN.BETA2 = 0.999
cfg.TRAIN.LR_TYPR = 'stone'
cfg.TRAIN.LR_REDUCE = [80,110]
cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.NUM_WORKERS = 16
cfg.TRAIN.USE_AUG = True
cfg.TRAIN.CROP = 448
cfg.TRAIN.PAD = 0
cfg.TRAIN.RESIZE = 300



cfg.MODEL = edict()
cfg.MODEL.NAME = 'resnet'
cfg.MODEL.BLOCK = 'bottleneck'
cfg.MODEL.BLOCK_LIST = [3,4,6,3]
cfg.MODEL.IN_DIM = 3
cfg.MODEL.CLASS_NUM = 200 
cfg.MODEL.BASE = 64 
cfg.MODEL.USE_FC = True 
cfg.MODEL.CONV1 = (7,2,3)
cfg.MODEL.OPERATION = 'B'
cfg.MODEL.STRIDE1 = 1
cfg.MODEL.MAX_POOL = True 
cfg.MODEL.PRETRAINED = False
cfg.MODEL.DROPOUT = 0 

cfg.MODEL.LOSS = 'bce_only_g' 

cfg.GPUS = [0]
cfg.PRINT_FRE = 50
cfg.DATASET_TRPE = 'cub200_2011'
cfg.SHORT_TEST = False


if __name__ == "__main__":
    from utils import load_cfg
    logger = load_cfg(cfg)
    print(cfg)




