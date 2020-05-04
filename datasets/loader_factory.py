import numpy as np
import torchvision.transforms as transforms
import torch
import cv2
from . import cifar


LOADER_LUT = {
        'cifar' : cifar.CIFARData,
    }


def get_loader(dataset_type, data_path, loader_type, label_path=None, cfg=None, logger=None):
    if loader_type == 'train':
        if cfg.USE_AUG == True:
            train_aug = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(cfg.CROP, cfg.PAD),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            train_aug = None

        try: 
            _data_class = LOADER_LUT.get(dataset_type)    
        except:
            logger.error("dataset type error, {} not exist".format(dataset_type))

        _data = _data_class(data_path,  dtype='train', label_path=label_path, aug=train_aug) 
        data_loader = torch.utils.data.DataLoader(_data,
            batch_size=cfg.BATCHSIZE, shuffle=True, num_workers=cfg.NUM_WORKERS,
            drop_last=False)

    elif loader_type == 'eval':
        if cfg.USE_AUG == True:
            val_aug = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        else:
            val_aug = None
        try: 
            _data_class = LOADER_LUT.get(dataset_type)    
        except:
            logger.error("dataset type error, {} not exist".format(dataset_type))

        _data = _data_class(data_path,  dtype='eval', label_path=label_path, aug=val_aug) 
        data_loader = torch.utils.data.DataLoader(_data,
            batch_size=128, shuffle=False, num_workers=16,
            drop_last=False)

    elif loader_type == 'self_test':
        augmentation = None
        try: 
            _data_class = LOADER_LUT.get(dataset_type)    
        except:
            logger.error("dataset type error, {} not exist".format(dataset_type))
        _data = _data_class(data_path, aug=augmentation) 
        data_loader = torch.utils.data.DataLoader(_data,
        batch_size=2, shuffle=True, num_workers=0,
        drop_last=True)
        
    elif loader_type == 'test':
        augmentaiton = transforms.Compose([
            transforms.ToTensor()
        ])
        #augmentaiton = None
        try: 
            _data_class = LOADER_LUT.get(dataset_type)    
        except:
            logger.error("dataset type error, {} not exist".format(dataset_type))
        _data = _data_class(data_path, aug=augmentaiton, test_data=True)
        data_loader = torch.utils.data.DataLoader(_data,
        batch_size=1, shuffle=False, num_workers=0,
        drop_last=False)

    else:
        logger.error("error, only support train type dataloader")
    
    return data_loader

 
def inverse_preprocess(image):
    image = image.numpy().transpose((1,2,0)) * 255
    image = image.astype(np.uint8)
    if image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
        #pass
    return image


def test_inpaint():
    import matplotlib.pyplot as plt 
    import random
    
    
    random.seed(0)
    torch.manual_seed(0)
    
    train_loader = get_loader("vsdr_inpaint", 
       './imgs/dataset/VSDR/train_data/291/',
        "self_test")
    for i, (img_ori_tensor, img_inpaint_tensor) in enumerate(train_loader):
        img_ori_tensor_0 = inverse_preprocess(img_ori_tensor[0])
        img_inpaint_tensor_0 = inverse_preprocess(img_inpaint_tensor[0])
        img_ori_tensor_1 = inverse_preprocess(img_ori_tensor[1])
        img_inpaint_tensor_1 = inverse_preprocess(img_inpaint_tensor[1])
       
        fig = plt.figure()
        a = fig.add_subplot(2,2,1)
        a.set_title('img_ori_tensor_0')
        plt.imshow(img_ori_tensor_0)
        a = fig.add_subplot(2,2,2)
        a.set_title('img_inpaint_tensor_0')
        plt.imshow(img_inpaint_tensor_0)
        a = fig.add_subplot(2,2,3)
        a.set_title('img_ori_tensor_1')
        plt.imshow(img_ori_tensor_1)
        a = fig.add_subplot(2,2,4)
        a.set_title('img_inpaint_tensor_1')
        plt.imshow(img_inpaint_tensor_1)
        plt.show()


if __name__ == "__main__":
    test_face()