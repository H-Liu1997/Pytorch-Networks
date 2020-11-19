import os
from easydict import EasyDict as edict



# 'data_path': "/kaggle/input/lyft-motion-prediction-autonomous-vehicles",
#     'model_params': {
#         'model_architecture': 'resnet34',
#         'history_num_frames': 10,
#         'history_step_size': 1,
#         'history_delta_time': 0.1,
#         'future_num_frames': 50,
#         'future_step_size': 1,
#         'future_delta_time': 0.1,
#         'model_name': "model_resnet34_output",
#         'lr': 1e-3,
#         'weight_path': "/kaggle/input/lyft-pretrained-model-hv/model_multi_update_lyft_public.pth",
#         'train': False,
#         'predict': True
#     },

#     'raster_params': {
#         'raster_size': [224, 224],
#         'pixel_size': [0.5, 0.5],
#         'ego_center': [0.25, 0.5],
#         'map_type': 'py_semantic',
#         'satellite_map_key': 'aerial_map/aerial_map.png',
#         'semantic_map_key': 'semantic_map/semantic_map.pb',
#         'dataset_meta_key': 'meta.json',
#         'filter_agents_threshold': 0.5
#     },

#     'train_data_loader': {
#         'key': 'scenes/train.zarr',
#         'batch_size': 16,
#         'shuffle': True,
#         'num_workers': 4
#     },
    
#     'test_data_loader': {
#         'key': 'scenes/test.zarr',
#         'batch_size': 32,
#         'shuffle': False,
#         'num_workers': 4
#     },

#     'train_params': {
#         'max_num_steps': 101,
#         'checkpoint_every_n_steps': 20,
#     }
# }



cfg = edict()
cfg.PATH = edict()
cfg.PATH.DATA =  'C:/Users/l84179161/Desktop/intern/Datasets/lyft-motion-prediction-autonomous-vehicles'
cfg.PATH.RES_TEST = './res_imgs/'
cfg.PATH.EXPS = './exps/'
cfg.PATH.NAME = 'reg32_cos_lyft_test'
cfg.PATH.MODEL = '/model.pth'
cfg.PATH.BESTMODEL = '/bestmodel.pth'
cfg.PATH.LOG = '/log.txt'
cfg.PATH.RESULTS = '/results/'


cfg.model_params = {
        'history_num_frames': 10,
        'history_step_size': 1,
        'history_delta_time': 0.1,
        'future_num_frames': 50,
        'future_step_size': 1,
        'future_delta_time': 0.1,}

cfg.raster_params = {
        'raster_size': [224, 224],
        'pixel_size': [0.5, 0.5],
        'ego_center': [0.25, 0.5],
        'map_type': 'py_semantic',
        'satellite_map_key': 'aerial_map/aerial_map.png',
        'semantic_map_key': 'semantic_map/semantic_map.pb',
        'dataset_meta_key': 'meta.json',
        'filter_agents_threshold': 0.5}

cfg.train_data_loader = {
        'key': 'scenes/train.zarr',
        'batch_size': 1,
        'shuffle': True,
        'num_workers': 1}


cfg.DETERMINISTIC = edict()
cfg.DETERMINISTIC.SEED = 60
cfg.DETERMINISTIC.CUDNN = True


cfg.TRAIN = edict()
cfg.TRAIN.EPOCHS = 1
#cfg.TRAIN.BATCHSIZE = 8
#cfg.TRAIN.L1SCALING = 100
cfg.TRAIN.TYPE = 'adam'
cfg.TRAIN.LR = 3e-4
cfg.TRAIN.BETA1 = 0.9
cfg.TRAIN.BETA2 = 0.999
cfg.TRAIN.LR_TYPE = 'stone'
cfg.TRAIN.LR_REDUCE = [0.6,0.8]
cfg.TRAIN.LR_FACTOR = 0.1
cfg.TRAIN.WEIGHT_DECAY = 0
#cfg.TRAIN.NUM_WORKERS = 16
cfg.TRAIN.WARMUP = 0
cfg.TRAIN.LR_WARM = 1e-7
#-------- data aug --------#
cfg.TRAIN.USE_AUG = True
cfg.TRAIN.CROP = 224
cfg.TRAIN.PAD = 0
cfg.TRAIN.RESIZE = 300
cfg.TRAIN.ROATION = 30


cfg.MODEL = edict()
cfg.MODEL.NAME = 'regnet'
cfg.MODEL.IN_DIM = 25
cfg.MODEL.CLASS_NUM = 200 #this is meaningless
cfg.MODEL.USE_FC = True
cfg.MODEL.PRETRAIN = 'RegNetY-3.2GF'
cfg.MODEL.PRETRAIN_PATH = 'C:/Users/l84179161/Desktop/intern/Datasets/Pretrain/'
cfg.MODEL.DROPOUT = 0
cfg.MODEL.LOSS = 'nll_loss' 
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
cfg.MODEL.REGNET.DEPTH = 21
cfg.MODEL.REGNET.W0 = 80
cfg.MODEL.REGNET.WA = 42.63
cfg.MODEL.REGNET.WM = 2.66
cfg.MODEL.REGNET.GROUP_W = 24
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

cfg.DATASET_TRPE = 'lyft'
cfg.SHORT_TEST = False


if __name__ == "__main__":
    from utils import load_cfg
    logger = load_cfg(cfg)
    print(cfg)




