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
cfg.PATH.NAME = 'rest_cub_v3_stone'
cfg.PATH.MODEL = '/model.pth'
cfg.PATH.BESTMODEL = '/bestmodel.pth'
cfg.PATH.LOG = '/log.txt'
cfg.PATH.RESULTS = '/results/'


cfg.DETERMINISTIC = edict()
cfg.DETERMINISTIC.SEED = 60
cfg.DETERMINISTIC.CUDNN = True


cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 40
cfg.TRAIN.BATCHSIZE = 8
cfg.TRAIN.L1SCALING = 100
cfg.TRAIN.TYPE = 'sgd'
cfg.TRAIN.LR = 1e-3
cfg.TRAIN.BETA1 = 0.9
cfg.TRAIN.BETA2 = 0.999
cfg.TRAIN.LR_TYPE = 'stone'
cfg.TRAIN.LR_REDUCE = [26,36]
cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.WEIGHT_DECAY = 1e-4
cfg.TRAIN.NUM_WORKERS = 16
cfg.TRAIN.WARMUP = 0
cfg.TRAIN.LR_WARM = 1e-7
#-------- data aug --------#
cfg.TRAIN.USE_AUG = True
cfg.TRAIN.CROP = 224
cfg.TRAIN.PAD = 0
cfg.TRAIN.RESIZE = 300
cfg.TRAIN.ROATION = 30


cfg.MODEL = edict()
cfg.MODEL.NAME = 'resnext'
cfg.MODEL.IN_DIM = 3
cfg.MODEL.CLASS_NUM = 200 
cfg.MODEL.USE_FC = True
cfg.MODEL.PRETRAIN = 'ResNeXt-50'
cfg.MODEL.PRETRAIN_PATH = './exps/pretrain/'
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
cfg.MODEL.REGNET.SE_ON = True
cfg.MODEL.REGNET.SE_R = 0.25
cfg.MODEL.REGNET.BOT_MUL = 1.0
cfg.MODEL.REGNET.DEPTH = 17
cfg.MODEL.REGNET.W0 = 192
cfg.MODEL.REGNET.WA = 76.82
cfg.MODEL.REGNET.WM = 2.19
cfg.MODEL.REGNET.GROUP_W = 56
#-------- for anynet -------#
cfg.MODEL.ANYNET = edict()
cfg.MODEL.ANYNET.STEM_TYPE = "res_stem_in"
cfg.MODEL.ANYNET.STEM_W = 64
cfg.MODEL.ANYNET.BLOCK_TYPE = "res_bottleneck_block"
cfg.MODEL.ANYNET.STRIDES = [1,2,2,2]
cfg.MODEL.ANYNET.SE_ON = False
cfg.MODEL.ANYNET.SE_R = 0.25
cfg.MODEL.ANYNET.BOT_MULS = [0.5,0.5,0.5,0.5]
cfg.MODEL.ANYNET.DEPTHS = [3,4,6,3]
cfg.MODEL.ANYNET.GROUP_WS = [4,8,16,32]
cfg.MODEL.ANYNET.WIDTHS = [256,512,1024,2048]
#-------- for effnet --------#
cfg.MODEL.EFFNET = edict()
cfg.MODEL.EFFNET.STEM_W = 32
cfg.MODEL.EFFNET.EXP_RATIOS = [1,6,6,6,6,6,6]
cfg.MODEL.EFFNET.KERNELS = [3,3,5,3,5,5,3]
cfg.MODEL.EFFNET.HEAD_W = 1408
cfg.MODEL.EFFNET.DC_RATIO = 0.0
cfg.MODEL.EFFNET.STRIDES = [1,2,2,2,1,2,1]
cfg.MODEL.EFFNET.SE_R = 0.25
cfg.MODEL.EFFNET.DEPTHS = [2, 3, 3, 4, 4, 5, 2]
cfg.MODEL.EFFNET.GROUP_WS = [4,8,16,32]
cfg.MODEL.EFFNET.WIDTHS = [16,24,48,88,120,208,352]

cfg.GPUS = [0]
cfg.PRINT_FRE = 300
cfg.DATASET_TRPE = 'cub200_2011'
cfg.SHORT_TEST = False


if __name__ == "__main__":
    from utils import load_cfg
    logger = load_cfg(cfg)
    print(cfg)




