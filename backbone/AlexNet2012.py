# --------------------------------------------------------------------------- #
# AlexNet, NIPS2012, https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['AlexNet']


class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet,self).__init__()
        self.net_layers=nn.Sequential(
            nn.Conv2d(3,96,11,4,0),
            nn.ReLU(),
            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(384,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,256,3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Flatten(),
            nn.Linear(9216,4096),
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
        for layer in self.net_layers:
            if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                print('init layer:',_flag,' weight success')
                if _flag == 1 or _flag == 3:
                    nn.init.constant_(layer.bias, 0.0)
                    print('init layer:',_flag,'bias to 0')
                else:
                    nn.init.constant_(layer.bias, 1.0)
                    print('init layer:',_flag,'bias to 1')
                _flag += 1
                
    def forward(self,input_):
        x = self.net_layers(input_)
        return x       


def _test():
    from torchsummary import summary
    model = AlexNet()
    model = model.cuda()
    summary(model,input_size=(3,224,224))
    

if __name__ == "__main__":
    _test()

# ------------------------------- mistakes ---------------------------------- #
# nn.Softmax(dim=1)
# ---------------------------------- end ------------------------------------ #



# ------------------------------- background -------------------------------- #
# 2010 ImageNet LSVRC-2010 Top-1 37.5% Top-5 17.0%
# 2012 ImageNet LSVRC-2012             Top-5 15.3%
# ImageNet 1200,000+train, 50,000+val, 150,000+test, 1000class
# ---------------------------------- end ------------------------------------ #


# ---------------------------------- notes ---------------------------------- #
# parameters: 60M + dropout0.5 + ReLU
# sgd+momentum 0.9
# batch size 128
# weight decay 5e-4
# special init 
# input: resize and crop samll side to 256×256 then augment to 224
# output: linear 1000 + softmax
# test: 10×TTA flip,center+4corner,average result, 7 model embedding in paper
# TODO: Check PCA in data augment, LRN, Overlapping Pooling
# TODO: original paper using 2 gpu, check connection.
# TODO: our second layer is 54, original 55.
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------------- model summary ----------------------------- #
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1           [-1, 96, 54, 54]          34,944
#               ReLU-2           [-1, 96, 54, 54]               0
#             Conv2d-3          [-1, 256, 54, 54]         614,656
#               ReLU-4          [-1, 256, 54, 54]               0
#          MaxPool2d-5          [-1, 256, 27, 27]               0
#             Conv2d-6          [-1, 384, 27, 27]         885,120
#               ReLU-7          [-1, 384, 27, 27]               0
#          MaxPool2d-8          [-1, 384, 13, 13]               0
#             Conv2d-9          [-1, 384, 13, 13]       1,327,488
#              ReLU-10          [-1, 384, 13, 13]               0
#            Conv2d-11          [-1, 256, 13, 13]         884,992
#              ReLU-12          [-1, 256, 13, 13]               0
#         MaxPool2d-13            [-1, 256, 6, 6]               0
#           Flatten-14                 [-1, 9216]               0
#            Linear-15                 [-1, 4096]      37,752,832
#              ReLU-16                 [-1, 4096]               0
#           Dropout-17                 [-1, 4096]               0
#            Linear-18                 [-1, 4096]      16,781,312
#              ReLU-19                 [-1, 4096]               0
#           Dropout-20                 [-1, 4096]               0
#            Linear-21                 [-1, 1000]       4,097,000
#           Softmax-22                 [-1, 1000]               0
# ================================================================
# Total params: 62,378,344
# Trainable params: 62,378,344
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 23.85
# Params size (MB): 237.95
# Estimated Total Size (MB): 262.38
# ---------------------------------- end ------------------------------------ #