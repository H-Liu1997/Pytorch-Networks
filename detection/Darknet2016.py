# --------------------------------------------------------------------------- #
# DarkNet, backbone for YOLOv2 https://arxiv.org/abs/1612.08242
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DarkNet_19']


class DarkNet_19(nn.Module):
    def __init__(self):
        super(DarkNet_19,self).__init__()
        self.net_layers=nn.Sequential(
            nn.Conv2d(3,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Conv2d(32,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,64,1,1,0),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,128,3,1,1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,128,1,1,0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128,256,3,1,1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,256,1,1,0),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            
            nn.Conv2d(512,1024,3,1,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1024,3,1,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024,512,3,1,1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512,1024,3,1,1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool2d((1,1)),

            nn.Flatten(),
            nn.Linear(1024,1000),
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
    model = DarkNet_19()
    torch.cuda.set_device(1)
    model = model.cuda()
    summary(model,input_size=(3,224,224))
    

if __name__ == "__main__":
    _test()



# ------------------------------- model summary ----------------------------- #
#              ReLU-47           [-1, 1024, 7, 7]               0
#            Conv2d-48            [-1, 512, 7, 7]       4,719,104
#       BatchNorm2d-49            [-1, 512, 7, 7]           1,024
#              ReLU-50            [-1, 512, 7, 7]               0
#            Conv2d-51           [-1, 1024, 7, 7]       4,719,616
#       BatchNorm2d-52           [-1, 1024, 7, 7]           2,048
#              ReLU-53           [-1, 1024, 7, 7]               0
#            Conv2d-54            [-1, 512, 7, 7]       4,719,104
#       BatchNorm2d-55            [-1, 512, 7, 7]           1,024
#              ReLU-56            [-1, 512, 7, 7]               0
#            Conv2d-57           [-1, 1024, 7, 7]       4,719,616
#       BatchNorm2d-58           [-1, 1024, 7, 7]           2,048
#              ReLU-59           [-1, 1024, 7, 7]               0
# AdaptiveAvgPool2d-60           [-1, 1024, 1, 1]               0
#           Flatten-61                 [-1, 1024]               0
#            Linear-62                 [-1, 1000]       1,025,000
#           Softmax-63                 [-1, 1000]               0
# ================================================================
# Total params: 29,238,184
# Trainable params: 29,238,184
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 109.32
# Params size (MB): 111.53
# Estimated Total Size (MB): 221.43
# ---------------------------------- end ------------------------------------ #