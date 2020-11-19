import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader
import cv2
import os

import l5kit
from l5kit.configs import load_config_data
from l5kit.data import LocalDataManager, ChunkedDataset
from l5kit.dataset import AgentDataset, EgoDataset
from l5kit.rasterization import build_rasterizer
from l5kit.evaluation import write_pred_csv, compute_metrics_csv, read_gt_csv, create_chopped_dataset
from l5kit.evaluation.chop_dataset import MIN_FUTURE_STEPS
from l5kit.evaluation.metrics import neg_multi_log_likelihood, time_displace


def get_loader(dataset_type, data_path, loader_type, label_path=None, cfg=None, logger=None):
    '''
    lyft type will not be used in this function, but it's an input
    '''

    if loader_type == 'train':
        # if cfg.USE_AUG == True:
        #     train_aug = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.RandomHorizontalFlip(),
        #         transforms.RandomRotation(cfg.ROATION),
        #         #transforms.RandomCrop(cfg.CROP, cfg.PAD),
        #         transforms.RandomResizedCrop(cfg.CROP),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     ])
        # else:
        #     train_aug = None

        os.environ["L5KIT_DATA_FOLDER"] = data_path
        dm = LocalDataManager(None)            
        train_cfg = cfg["train_data_loader"]
        rasterizer = build_rasterizer(cfg, dm)
        train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
        data_loader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
                                    num_workers=train_cfg["num_workers"])
        logger.info(train_dataset) 

    elif loader_type == 'eval':
        pass
        # if cfg.USE_AUG == True:
        #     val_aug = transforms.Compose([
        #         transforms.ToPILImage(),
        #         transforms.Resize(int(cfg.CROP/0.875)),
        #         transforms.CenterCrop(cfg.CROP),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     ])
        # else:
        #     val_aug = None
        # os.environ["L5KIT_DATA_FOLDER"] = data_path
        # dm = LocalDataManager(None)
        # train_cfg = cfg["train_data_loader"]
        # rasterizer = build_rasterizer(cfg, dm)
        # train_zarr = ChunkedDataset(dm.require(train_cfg["key"])).open()
        # train_dataset = AgentDataset(cfg, train_zarr, rasterizer)
        # train_dataloader = DataLoader(train_dataset, shuffle=train_cfg["shuffle"], batch_size=train_cfg["batch_size"], 
        #                             num_workers=train_cfg["num_workers"])
        # logger.info(train_dataset)
        
    elif loader_type == 'self_test':
        pass
        # augmentation = transforms.Compose([
        #         transforms.ToTensor(),
        #         #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        #     ])
        # try: 
        #     _data_class = LOADER_LUT.get(dataset_type)    
        # except:
        #     logger.error("dataset type error, {} not exist".format(dataset_type))
        # _data = _data_class(data_path,  dtype='train', label_path=label_path, aug=augmentation, cfg=cfg) 
        # data_loader = torch.utils.data.DataLoader(_data,
        # batch_size=2, shuffle=True, num_workers=0,
        # drop_last=True)
        
    elif loader_type == 'test':
        pass
        # augmentaiton = transforms.Compose([
        #     transforms.ToTensor()
        # ])
        # #augmentaiton = None
        # try: 
        #     _data_class = LOADER_LUT.get(dataset_type)    
        # except:
        #     logger.error("dataset type error, {} not exist".format(dataset_type))
        # _data = _data_class(data_path, aug=augmentaiton, test_data=True)
        # data_loader = torch.utils.data.DataLoader(_data,
        # batch_size=1, shuffle=False, num_workers=0,
        # drop_last=False)

    else:
        logger.error(f"error, {loader_type} is not exist")
    
    return data_loader

 
def inverse_preprocess(image):
    image = image.numpy().transpose((1,2,0)) * 255
    image = image.astype(np.uint8)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        #pass
    return image


def test_():
   from config_lyft import cfg
   train_loader = get_loader(cfg.DATASET_TRPE, cfg.PATH.DATA, 'train', label_path=None, cfg=cfg, logger=None)




if __name__ == "__main__":
    test_()