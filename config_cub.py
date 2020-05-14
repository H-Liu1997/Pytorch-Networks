import os
from easydict import EasyDict as edict


cfg = edict()
cfg.PATH = edict()
cfg.PATH.DATA =  ['/home/liuhaiyang/dataset/CUB_200_2011/images.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/train_test_split.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/images/']
cfg.PATH.LABEL = '/home/liuhaiyang/dataset/CUB_200_2011/image_class_labels.txt'
cfg.PATH.EVAL = ['/home/liuhaiyang/dataset/CUB_200_2011/images.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/train_test_split.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/images/']
cfg.PATH.TEST = '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_1'
cfg.PATH.RES_TEST = './res_imgs/'
cfg.PATH.EXPS = './exps/'
cfg.PATH.NAME = 'reg_50_cub_v7_cos'
cfg.PATH.MODEL = '/model.pth'
cfg.PATH.BESTMODEL = '/bestmodel.pth'
cfg.PATH.LOG = '/log.txt'
cfg.PATH.RESULTS = '/results/'


cfg.DETERMINISTIC = edict()
cfg.DETERMINISTIC.SEED = 60
cfg.DETERMINISTIC.CUDNN = True


cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 42
cfg.TRAIN.BATCHSIZE = 8
cfg.TRAIN.L1SCALING = 100
cfg.TRAIN.TYPE = 'sgd'
cfg.TRAIN.LR = 1e-3
cfg.TRAIN.BETA1 = 0.9
cfg.TRAIN.BETA2 = 0.999
cfg.TRAIN.LR_TYPE = 'cos'
cfg.TRAIN.LR_REDUCE = [29,36]
cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.NUM_WORKERS = 16
cfg.TRAIN.WARMUP = 2
cfg.TRAIN.LR_WARM = 1e-6
#-------- data aug --------#
cfg.TRAIN.USE_AUG = True
cfg.TRAIN.CROP = 224
cfg.TRAIN.PAD = 0
cfg.TRAIN.RESIZE = 300
cfg.TRAIN.ROATION = 30


cfg.MODEL = edict()
cfg.MODEL.NAME = 'resnet'
cfg.MODEL.IN_DIM = 3
cfg.MODEL.CLASS_NUM = 200 
cfg.MODEL.USE_FC = True
cfg.MODEL.PRETRAINED = False
cfg.MODEL.DROPOUT = 0
cfg.MODEL.LOSS = 'bce_only_g' 
#-------- for resnet --------#
cfg.MODEL.BLOCK = 'bottleneck'
cfg.MODEL.BLOCK_LIST = [3,4,6,3] 
cfg.MODEL.CONV1 = (7,2,3)
cfg.MODEL.OPERATION = 'B'
cfg.MODEL.STRIDE1 = 1
cfg.MODEL.MAX_POOL = True
cfg.MODEL.BASE = 64
#-------- for regnet --------#
cfg.MODEL.REGNET = edict()
cfg.MODEL.REGNET.STEM_TYPE = "simple_stem_in"
cfg.MODEL.REGNET.STEM_W = 32
cfg.MODEL.REGNET.BLOCK_TYPE = "res_bottleneck_block"
cfg.MODEL.REGNET.STRIDE = 2
cfg.MODEL.REGNET.SE_ON = False
cfg.MODEL.REGNET.SE_R = 0.25
cfg.MODEL.REGNET.BOT_MUL = 1.0
cfg.MODEL.REGNET.DEPTH = 23
cfg.MODEL.REGNET.W0 = 96
cfg.MODEL.REGNET.WA = 38.65
cfg.MODEL.REGNET.WM = 2.43
cfg.MODEL.REGNET.GROUP_W = 40
#-------- for anynet -------#
cfg.MODEL.ANYNET = edict()
cfg.MODEL.ANYNET.STEM_TYPE = "simple_stem_in"
cfg.MODEL.ANYNET.STEM_W = 32
cfg.MODEL.ANYNET.BLOCK_TYPE = "res_bottleneck_block"
cfg.MODEL.ANYNET.STRIDES = []
cfg.MODEL.ANYNET.SE_ON = False
cfg.MODEL.ANYNET.SE_R = 0.25
cfg.MODEL.ANYNET.BOT_MULS = []
cfg.MODEL.ANYNET.DEPTHS = []
cfg.MODEL.ANYNET.GROUP_WS = []


cfg.GPUS = [0]
cfg.PRINT_FRE = 300
cfg.DATASET_TRPE = 'cub200_2011'
cfg.SHORT_TEST = False


if __name__ == "__main__":
    from utils import load_cfg
    logger = load_cfg(cfg)
    print(cfg)




