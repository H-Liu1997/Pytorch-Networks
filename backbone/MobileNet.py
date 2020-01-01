# --------------------------------------------------------------------------- #
# MobileNet_v1, https://arxiv.org/abs/1704.04861
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MobileNet_V1']


class BasicConv(nn.Sequential):
    def __init__(self,in_dim,out_dim,k,s,p):
        super(BasicConv,self).__init__()
        self.conv = nn.Conv2d(in_dim,out_dim,k,s,p,bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.relu = nn.ReLU(inplace=True)
    
class DPConv(nn.Sequential):
    def __init__(self,in_dim,out_dim,stride):
        super(DPConv,self).__init__()
        self.conv_1 = nn.Conv2d(in_dim,in_dim,3,stride,1,bias=False,groups=in_dim)
        self.bn1 = nn.BatchNorm2d(in_dim)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_2 = nn.Conv2d(in_dim,out_dim,1,1,0,bias=False)
        self.bn2 = nn.BatchNorm2d(out_dim)
        self.relu2 = nn.ReLU(inplace =True)
        

class MobileNet_V1(nn.Sequential):
    def __init__(self,):
        super(MobileNet_V1,self).__init__()
        self.conv = nn.Sequential(BasicConv(3,32,3,2,1),
             DPConv(32,64,1),
             DPConv(64,128,2),
             DPConv(128,128,1),
             DPConv(128,256,2),
             DPConv(256,256,1),
             DPConv(256,512,2),

             DPConv(512,512,1),
             DPConv(512,512,1),
             DPConv(512,512,1),
             DPConv(512,512,1),
             DPConv(512,512,1),

             DPConv(512,1024,2),
             DPConv(1024,1024,1),)
        
        self.final = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(1024,1000),
            nn.Softmax(dim=1)
        )


def _test():
    from torchsummary import summary
    model = MobileNet_V1()
    torch.cuda.set_device(1)
    model = model.cuda()
    summary(model,input_size=(3,224,224))
    

if __name__ == "__main__":
    _test()

# ------------------------------- model summary ----------------------------- #
#              ReLU-69          [-1, 512, 14, 14]               0
#            Conv2d-70            [-1, 512, 7, 7]           4,608
#       BatchNorm2d-71            [-1, 512, 7, 7]           1,024
#              ReLU-72            [-1, 512, 7, 7]               0
#            Conv2d-73           [-1, 1024, 7, 7]         524,288
#       BatchNorm2d-74           [-1, 1024, 7, 7]           2,048
#              ReLU-75           [-1, 1024, 7, 7]               0
#            Conv2d-76           [-1, 1024, 7, 7]           9,216
#       BatchNorm2d-77           [-1, 1024, 7, 7]           2,048
#              ReLU-78           [-1, 1024, 7, 7]               0
#            Conv2d-79           [-1, 1024, 7, 7]       1,048,576
#       BatchNorm2d-80           [-1, 1024, 7, 7]           2,048
#              ReLU-81           [-1, 1024, 7, 7]               0
# AdaptiveAvgPool2d-82           [-1, 1024, 1, 1]               0
#           Flatten-83                 [-1, 1024]               0
#            Linear-84                 [-1, 1000]       1,025,000
#           Softmax-85                 [-1, 1000]               0
# ================================================================
# Total params: 4,231,976
# Trainable params: 4,231,976
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 115.45
# Params size (MB): 16.14
# Estimated Total Size (MB): 132.17
# ---------------------------------- end ------------------------------------ #