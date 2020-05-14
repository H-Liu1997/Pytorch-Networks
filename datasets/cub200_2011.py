import os
import numpy as np 
import torch 
import cv2
import math
from random import randint
import torchvision.transforms as transforms


class CUBData(torch.utils.data.Dataset):
    def __init__(self, img_path, dtype='train', label_path=None, aug=None, cfg=None):
        self.data_list = []
        self.train_or_test = []
        self.label = []
        self.dtype = dtype
        self.aug = aug
        self.img_size = cfg

        with open(img_path[0],'r') as f:
            for line in f:
                sub_path = line.split(" ")[1]
                self.data_list.append(img_path[2]+sub_path)
        
        with open(img_path[1],'r') as f:
            if dtype == 'train':
                for line in f:
                    id_type = line.split(" ")
                    if int(id_type[1]) == 1:
                        self.train_or_test.append(int(id_type[0]))
            elif dtype == 'eval':
                for line in f:
                    id_type = line.split(" ")
                    if int(id_type[1]) == 0:
                        self.train_or_test.append(int(id_type[0]))

        with open(label_path, 'r') as f:
            for line in f:
                self.label.append(int(line.split(" ")[1])-1)
       

    def _resize(self, img_data):
        large_edge = max(img_data.shape[0], img_data.shape[1])
        resize_factor = self.img_size/large_edge
        zero_data = np.zeros((self.img_size, self.img_size,3))
        img_data = cv2.resize(img_data, (min(math.ceil(img_data.shape[1]*resize_factor),self.img_size),
            min(math.ceil(img_data.shape[0]*resize_factor),self.img_size)))
        if img_data.shape[0] >= img_data.shape[1]:
            gap = (self.img_size-img_data.shape[1])//2
            zero_data[:,gap:img_data.shape[1]+gap,:] = img_data
        else:
            gap = (self.img_size-img_data.shape[0])//2
            zero_data[gap:img_data.shape[0]+gap,:,:] = img_data
        return zero_data


    def __getitem__(self,index):
        img_data = cv2.imread(self.data_list[self.train_or_test[index]-1][:-1])
        img_data_resize = img_data#self._resize(img_data)
        if self.aug is not None:
            data_final = self.aug(img_data_resize.astype(np.uint8))
        if self.dtype == 'test':
            return data_final
        else:
            label_final = torch.from_numpy(np.array(self.label[self.train_or_test[index]-1]))
            return data_final,label_final

    def __len__(self):
        return len(self.train_or_test)