# --------------------------------------------------------------------------- #
# VGGNet, ICLR2015, https://arxiv.org/abs/1409.1556
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F

#vgg16 is conv3 version
__all__ = ['vgg16','vgg19']


class vgg16(nn.Module):
    def __init__(self):
        super(vgg16,self).__init__()
        self.net_layers=nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(25088,4096),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(4096,1000),
            nn.Softmax(dim=1),   
        )
        self.initialization()
    
    def initialization(self):
        _flag = 1 
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                print('init layer:',_flag,' weight success')
                nn.init.constant_(layer.bias, 0.0)
                print('init layer:',_flag,'bias to 0')
                _flag += 1
                
    def forward(self,input_):
        x = self.net_layers(input_)
        return x       

class vgg19(nn.Module):
    def __init__(self):
        super(vgg19,self).__init__()
        self.net_layers=nn.Sequential(
            nn.Conv2d(3,64,3,1,1),
            nn.ReLU(),
            nn.Conv2d(64,64,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,1,1),
            nn.ReLU(),
            nn.Conv2d(128,128,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.Conv2d(256,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.Conv2d(512,512,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(25088,4096),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(0.50),
            nn.Linear(4096,1000),
            nn.Softmax(dim=1),   
        )
        self.initialization()
    
    def initialization(self):
        _flag = 1 
        for layer in self.modules():
            if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                print('init layer:',_flag,' weight success')
                nn.init.constant_(layer.bias, 0.0)
                print('init layer:',_flag,'bias to 0')
                _flag += 1
                
    def forward(self,input_):
        #x = super(vgg19,self).forward(input_) cannot work
        x = self.net_layers(input_)
        return x

def _test():
    from torchsummary import summary
    model = vgg19()
    model = model.cuda()
    summary(model,input_size=(3,224,224))
    

if __name__ == "__main__":
    _test()


# ------------------------------- background -------------------------------- #
# 2014 ImageNet LSVRC-2014 Top-1 23.7%  Top-5 7.32% 2rd
# ImageNet 1200,000+train, 50,000+val, 150,000+test, 1000class
# ---------------------------------- end ------------------------------------ #


# ---------------------------------- notes ---------------------------------- #
# main idea: several small kernal better than large kernel
# parameters: 138M vgg16, 144M vgg19, ReLU
# sgd+momentum 1e-2 0.9 divide 10 * 3 
# batch size 256
# weight decay 5e-4
# input: resize and crop samll side to 256×256 then augment to 224
# output: linear 1000 + softmax
# TODO: Check details in training,testing,such as first trianA
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# --------------------------- vgg16 model summary --------------------------- #
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 224, 224]           1,792
#               ReLU-2         [-1, 64, 224, 224]               0
#             Conv2d-3         [-1, 64, 224, 224]          36,928
#               ReLU-4         [-1, 64, 224, 224]               0
#          MaxPool2d-5         [-1, 64, 112, 112]               0
#             Conv2d-6        [-1, 128, 112, 112]          73,856
#               ReLU-7        [-1, 128, 112, 112]               0
#                                 ...
#              ReLU-30          [-1, 512, 14, 14]               0
#         MaxPool2d-31            [-1, 512, 7, 7]               0
#           Flatten-32                [-1, 25088]               0
#            Linear-33                 [-1, 4096]     102,764,544
#              ReLU-34                 [-1, 4096]               0
#           Dropout-35                 [-1, 4096]               0
#            Linear-36                 [-1, 4096]      16,781,312
#              ReLU-37                 [-1, 4096]               0
#           Dropout-38                 [-1, 4096]               0
#            Linear-39                 [-1, 1000]       4,097,000
#           Softmax-40                 [-1, 1000]               0
# ================================================================
# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 218.79
# Params size (MB): 527.79
# Estimated Total Size (MB): 747.16
# ---------------------------------- end ------------------------------------ #

