import os
from easydict import EasyDict as edict


cfg1 = edict()
cfg1.PATH = edict()
cfg1.PATH.DATA =  ['/home/liuhaiyang/dataset/CUB_200_2011/images.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/train_test_split.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/images/']
cfg1.PATH.LABEL = '/home/liuhaiyang/dataset/CUB_200_2011/image_class_labels.txt'
cfg1.PATH.EVAL = ['/home/liuhaiyang/dataset/CUB_200_2011/images.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/train_test_split.txt',
       '/home/liuhaiyang/dataset/CUB_200_2011/images/']
cfg1.PATH.TEST = '/home/liuhaiyang/liu_kaggle/cifar/dataset/cifar-10-batches-py/data_batch_1'
cfg1.PATH.RES_TEST = './res_imgs/'
cfg1.PATH.EXPS = './exps/'
cfg1.PATH.NAME = 'eff_cub_v2_stone'
cfg1.PATH.MODEL = '/model.pth'
cfg1.PATH.BESTMODEL = '/bestmodel.pth'
cfg1.PATH.LOG = '/log.txt'
cfg1.PATH.RESULTS = '/results/'


cfg1.DETERMINISTIC = edict()
cfg1.DETERMINISTIC.SEED = 60
cfg1.DETERMINISTIC.CUDNN = True


cfg1.TRAIN = edict()
cfg1.TRAIN.EPOCHS = 60
cfg1.TRAIN.BATCHSIZE = 8
cfg1.TRAIN.L1SCALING = 100
cfg1.TRAIN.TYPE = 'sgd'
cfg1.TRAIN.LR = 1e-3
cfg1.TRAIN.BETA1 = 0.9
cfg1.TRAIN.BETA2 = 0.999
cfg1.TRAIN.LR_TYPE = 'cos'
cfg1.TRAIN.LR_REDUCE = [26,36]
cfg1.TRAIN.LR_FACTOR = 0.1
cfg1.TRAIN.WEIGHT_DECAY = 1e-4
cfg1.TRAIN.NUM_WORKERS = 16
cfg1.TRAIN.WARMUP = 0
cfg1.TRAIN.LR_WARM = 1e-7
#-------- data aug --------#
cfg1.TRAIN.USE_AUG = True
cfg1.TRAIN.CROP = 224
cfg1.TRAIN.PAD = 0
cfg1.TRAIN.RESIZE = 300
cfg1.TRAIN.ROATION = 30


cfg1.MODEL = edict()
cfg1.MODEL.NAME = 'regnet'
cfg1.MODEL.IN_DIM = 3
cfg1.MODEL.CLASS_NUM = 200 
cfg1.MODEL.USE_FC = True
cfg1.MODEL.PRETRAIN = None
cfg1.MODEL.PRETRAIN_PATH = './exps/pretrain/'
cfg1.MODEL.DROPOUT = 0
cfg1.MODEL.LOSS = 'bce_only_g' 
#-------- for resnet --------#
cfg1.MODEL.BLOCK = 'bottleneck'
cfg1.MODEL.BLOCK_LIST = [3,4,6,3] 
cfg1.MODEL.CONV1 = (7,2,3)
cfg1.MODEL.OPERATION = 'B'
cfg1.MODEL.STRIDE1 = 1
cfg1.MODEL.MAX_POOL = True
cfg1.MODEL.BASE = 64
#-------- for regnet --------#
cfg1.MODEL.REGNET = edict()
cfg1.MODEL.REGNET.STEM_TYPE = "simple_stem_in"
cfg1.MODEL.REGNET.STEM_W = 32
cfg1.MODEL.REGNET.BLOCK_TYPE = "res_bottleneck_block"
cfg1.MODEL.REGNET.STRIDE = 2
cfg1.MODEL.REGNET.SE_ON = True
cfg1.MODEL.REGNET.SE_R = 0.25
cfg1.MODEL.REGNET.BOT_MUL = 1.0
cfg1.MODEL.REGNET.DEPTH = 20
cfg1.MODEL.REGNET.W0 = 232
cfg1.MODEL.REGNET.WA = 115.89
cfg1.MODEL.REGNET.WM = 2.53
cfg1.MODEL.REGNET.GROUP_W = 232
#-------- for anynet -------#
cfg1.MODEL.ANYNET = edict()
cfg1.MODEL.ANYNET.STEM_TYPE = "res_stem_in"
cfg1.MODEL.ANYNET.STEM_W = 64
cfg1.MODEL.ANYNET.BLOCK_TYPE = "res_bottleneck_block"
cfg1.MODEL.ANYNET.STRIDES = [1,2,2,2]
cfg1.MODEL.ANYNET.SE_ON = False
cfg1.MODEL.ANYNET.SE_R = 0.25
cfg1.MODEL.ANYNET.BOT_MULS = [0.5,0.5,0.5,0.5]
cfg1.MODEL.ANYNET.DEPTHS = [3,4,6,3]
cfg1.MODEL.ANYNET.GROUP_WS = [4,8,16,32]
cfg1.MODEL.ANYNET.WIDTHS = [256,512,1024,2048]
#-------- for effnet --------#
cfg1.MODEL.EFFNET = edict()
cfg1.MODEL.EFFNET.STEM_W = 32
cfg1.MODEL.EFFNET.EXP_RATIOS = [1,6,6,6,6,6,6]
cfg1.MODEL.EFFNET.KERNELS = [3,3,5,3,5,5,3]
cfg1.MODEL.EFFNET.HEAD_W = 1408
cfg1.MODEL.EFFNET.DC_RATIO = 0.0
cfg1.MODEL.EFFNET.STRIDES = [1,2,2,2,1,2,1]
cfg1.MODEL.EFFNET.SE_R = 0.25
cfg1.MODEL.EFFNET.DEPTHS = [2, 3, 3, 4, 4, 5, 2]
cfg1.MODEL.EFFNET.GROUP_WS = [4,8,16,32]
cfg1.MODEL.EFFNET.WIDTHS = [16,24,48,88,120,208,352]


cfg1.GPUS = [0]
cfg1.PRINT_FRE = 300

cfg1.DATASET_TRPE = 'cub200_2011'
cfg1.SHORT_TEST = False


if __name__ == "__main__":
    from utils import load_cfg1
    logger = load_cfg1(cfg1)
    print(cfg1)




