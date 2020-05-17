import os
from easydict import EasyDict as edict


cfg2 = edict()
cfg2.PATH = edict()
cfg2.PATH.DATA =  ['/home/liuhaiyang/dataset/CUB_200_2011/images.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/train_test_split.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/images/']
cfg2.PATH.LABEL = '/home/liuhaiyang/dataset/CUB_200_2011/image_class_labels.txt'
cfg2.PATH.EVAL = ['/home/liuhaiyang/dataset/CUB_200_2011/images.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/train_test_split.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/images/']
cfg2.PATH.TEST = '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_1'
cfg2.PATH.RES_TEST = './res_imgs/'
cfg2.PATH.EXPS = './exps/'
cfg2.PATH.NAME = 'rest_cub_v3_stone'
cfg2.PATH.MODEL = '/model.pth'
cfg2.PATH.BESTMODEL = '/bestmodel.pth'
cfg2.PATH.LOG = '/log.txt'
cfg2.PATH.RESULTS = '/results/'


cfg2.DETERMINISTIC = edict()
cfg2.DETERMINISTIC.SEED = 60
cfg2.DETERMINISTIC.CUDNN = True


cfg2.TRAIN = edict()
cfg2.TRAIN.EPOCHS = 60
cfg2.TRAIN.BATCHSIZE = 8
cfg2.TRAIN.L1SCALING = 100
cfg2.TRAIN.TYPE = 'sgd'
cfg2.TRAIN.LR = 1e-3
cfg2.TRAIN.BETA1 = 0.9
cfg2.TRAIN.BETA2 = 0.999
cfg2.TRAIN.LR_TYPE = 'cos'
cfg2.TRAIN.LR_REDUCE = [26,36]
cfg2.TRAIN.LR_FACTOR = 0.1
cfg2.TRAIN.WEIGHT_DECAY = 1e-4
cfg2.TRAIN.NUM_WORKERS = 16
cfg2.TRAIN.WARMUP = 0
cfg2.TRAIN.LR_WARM = 1e-7
#-------- data aug --------#
cfg2.TRAIN.USE_AUG = True
cfg2.TRAIN.CROP = 224
cfg2.TRAIN.PAD = 0
cfg2.TRAIN.RESIZE = 300
cfg2.TRAIN.ROATION = 30


cfg2.MODEL = edict()
cfg2.MODEL.NAME = 'resnext'
cfg2.MODEL.IN_DIM = 3
cfg2.MODEL.CLASS_NUM = 200 
cfg2.MODEL.USE_FC = True
cfg2.MODEL.PRETRAIN = None
cfg2.MODEL.PRETRAIN_PATH = './exps/pretrain/'
cfg2.MODEL.DROPOUT = 0
cfg2.MODEL.LOSS = 'bce_only_g' 
#-------- for resnet --------#
cfg2.MODEL.BLOCK = 'bottleneck'
cfg2.MODEL.BLOCK_LIST = [3,4,6,3] 
cfg2.MODEL.CONV1 = (7,2,3)
cfg2.MODEL.OPERATION = 'B'
cfg2.MODEL.STRIDE1 = 1
cfg2.MODEL.MAX_POOL = True
cfg2.MODEL.BASE = 64
#-------- for regnet --------#
cfg2.MODEL.REGNET = edict()
cfg2.MODEL.REGNET.STEM_TYPE = "simple_stem_in"
cfg2.MODEL.REGNET.STEM_W = 32
cfg2.MODEL.REGNET.BLOCK_TYPE = "res_bottleneck_block"
cfg2.MODEL.REGNET.STRIDE = 2
cfg2.MODEL.REGNET.SE_ON = True
cfg2.MODEL.REGNET.SE_R = 0.25
cfg2.MODEL.REGNET.BOT_MUL = 1.0
cfg2.MODEL.REGNET.DEPTH = 20
cfg2.MODEL.REGNET.W0 = 232
cfg2.MODEL.REGNET.WA = 115.89
cfg2.MODEL.REGNET.WM = 2.53
cfg2.MODEL.REGNET.GROUP_W = 232
#-------- for anynet -------#
cfg2.MODEL.ANYNET = edict()
cfg2.MODEL.ANYNET.STEM_TYPE = "res_stem_in"
cfg2.MODEL.ANYNET.STEM_W = 64
cfg2.MODEL.ANYNET.BLOCK_TYPE = "res_bottleneck_block"
cfg2.MODEL.ANYNET.STRIDES = [1,2,2,2]
cfg2.MODEL.ANYNET.SE_ON = False
cfg2.MODEL.ANYNET.SE_R = 0.25
cfg2.MODEL.ANYNET.BOT_MULS = [0.5,0.5,0.5,0.5]
cfg2.MODEL.ANYNET.DEPTHS = [3,4,6,3]
cfg2.MODEL.ANYNET.GROUP_WS = [4,8,16,32]
cfg2.MODEL.ANYNET.WIDTHS = [256,512,1024,2048]
#-------- for effnet --------#
cfg2.MODEL.EFFNET = edict()
cfg2.MODEL.EFFNET.STEM_W = 32
cfg2.MODEL.EFFNET.EXP_RATIOS = [1,6,6,6,6,6,6]
cfg2.MODEL.EFFNET.KERNELS = [3,3,5,3,5,5,3]
cfg2.MODEL.EFFNET.HEAD_W = 1408
cfg2.MODEL.EFFNET.DC_RATIO = 0.0
cfg2.MODEL.EFFNET.STRIDES = [1,2,2,2,1,2,1]
cfg2.MODEL.EFFNET.SE_R = 0.25
cfg2.MODEL.EFFNET.DEPTHS = [2, 3, 3, 4, 4, 5, 2]
cfg2.MODEL.EFFNET.GROUP_WS = [4,8,16,32]
cfg2.MODEL.EFFNET.WIDTHS = [16,24,48,88,120,208,352]


cfg2.GPUS = [0]
cfg2.PRINT_FRE = 300

cfg2.DATASET_TRPE = 'cub200_2011'
cfg2.SHORT_TEST = False


if __name__ == "__main__":
    from utils import load_cfg2
    logger = load_cfg2(cfg2)
    print(cfg2)




