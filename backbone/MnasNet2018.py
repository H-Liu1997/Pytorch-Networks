# --------------------------------------------------------------------------- #
# MnasNet, CVPR2019, https://arxiv.org/abs/1807.11626
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['MnasNet_A1']


class _SElayer(nn.Module):
    def __init__(self,in_dim,ratio):
        super(_SElayer,self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        reduced_dim = max(1, in_dim//ratio)
        self.fc1 = nn.Sequential(nn.Flatten(),
                   nn.Linear(in_dim, reduced_dim),
                   nn.ReLU(inplace=True),
                   nn.Linear(reduced_dim, in_dim),
                   nn.Softmax(dim=1),)

    def forward(self,input_):
        x_input = input_

        x = self.gap(input_)
        x = self.fc1(x)
        x = x.view(-1,x_input.shape[1],1,1)
        print(x.shape)
        x = x_input * x.expand_as(x_input) 
        return x

class _DWConv(nn.Sequential):
    def __init__(self,in_dim,k_size):
        super(_DWConv,self).__init__()
        self.add_module('depthwise',nn.Conv2d(in_dim,in_dim,k_size,1,
                       (k_size-1)//2,bias=False,groups=in_dim)),
        self.add_module('bn',nn.BatchNorm2d(in_dim)),
        self.add_module('relu',nn.ReLU(inplace=False))
        
class _Conv(nn.Sequential):
    def __init__(self,in_dim,out_dim,k,s,p):
        super(_Conv,self).__init__()
        self.add_module('conv',nn.Conv2d(in_dim,out_dim,k,s,p,bias=False)),
        self.add_module('bn',nn.BatchNorm2d(out_dim)),
        self.add_module('relu',nn.ReLU(inplace=True))


class _SepConv(nn.Sequential):
    def __init__(self,in_dim,out_dim,k):
        super(_SepConv,self).__init__()
        self.add_module('DWConv3×3',_DWConv(in_dim,3)),
        self.add_module('Conv1×1',_Conv(in_dim,out_dim,1,1,0))
    
class _MBConv(nn.Module):
    def __init__(self,in_dim,out_dim,depth,k,stride,SE_ratio=None):
        super(_MBConv,self).__init__()
        self.Conv1_1 = _Conv(in_dim,depth*in_dim,1,stride,0)
        self.DWConv3_1 = _DWConv(depth*in_dim,k)
        if stride != 1 or in_dim != out_dim:
            self.Downsample = _Conv(in_dim,out_dim,1,stride,0)
        else:
            self.Downsample = None
        if SE_ratio is not None:
            self.SElayer_ = _SElayer(depth*in_dim,SE_ratio)
        else:
            self.SElayer_ = None
        self.Conv1_2 = _Conv(depth*in_dim,out_dim,1,1,0)
        
    def forward(self,input_):
        x = self.Conv1_1(input_)
        if self.Downsample is not None:
            input_ = self.Downsample(input_)
        x = self.DWConv3_1(x)
        if self.SElayer_ is not None:
            x = self.SElayer_(x)
        x = self.Conv1_2(x)
        x += input_
        return x

class MnasNet_A1(nn.Sequential):
    def __init__(self):
        super(MnasNet_A1,self).__init__()
        self.HeadConv = _Conv(3,32,3,2,1)
        self.Seq_1 = _SepConv(32,16,3)

        self.MBConv6_1 = _MBConv(16,24,6,3,2)
        self.MBConv6_2 = _MBConv(24,24,6,3,1)

        self.MBConv3_1 = _MBConv(24,40,3,5,2,4)
        self.MBConv3_2 = _MBConv(40,40,3,5,1,4)
        self.MBConv3_3 = _MBConv(40,40,3,5,1,4)

        self.MBConv6_3 = _MBConv(40,80,6,3,2)
        self.MBConv6_4 = _MBConv(80,80,6,3,1)
        self.MBConv6_5 = _MBConv(80,80,6,3,1)
        self.MBConv6_6 = _MBConv(80,80,6,3,1)

        self.MBConv6_7 = _MBConv(80,112,6,3,1,4)
        self.MBConv6_8 = _MBConv(112,112,6,3,1,4)

        self.MBConv6_9 = _MBConv(112,160,6,5,2,4)
        self.MBConv6_10 = _MBConv(160,160,6,5,1,4)
        self.MBConv6_11 = _MBConv(160,160,6,5,1,4)
        
        self.MBConv6_12 = _MBConv(160,320,6,3,1)

        self.logits = nn.Sequential(
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(320,1000),
            nn.Softmax(dim=1))


def _test():
    from torchsummary import summary
    model = MnasNet_A1()
    model = model.cuda()
    summary(model,input_size=(3,224,224))

if __name__ == "__main__":
    _test()


# ---------------------------------- notes ---------------------------------- #
# main idea: new nas method
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------- MnasNet A1 model summary ------------------------ #
#           Conv2d-224            [-1, 320, 7, 7]          51,200
#      BatchNorm2d-225            [-1, 320, 7, 7]             640
#             ReLU-226            [-1, 320, 7, 7]               0
#           Conv2d-227            [-1, 960, 7, 7]           8,640
#      BatchNorm2d-228            [-1, 960, 7, 7]           1,920
#             ReLU-229            [-1, 960, 7, 7]               0
#           Conv2d-230            [-1, 320, 7, 7]         307,200
#      BatchNorm2d-231            [-1, 320, 7, 7]             640
#             ReLU-232            [-1, 320, 7, 7]               0
#          _MBConv-233            [-1, 320, 7, 7]               0
# AdaptiveAvgPool2d-234            [-1, 320, 1, 1]               0
#          Flatten-235                  [-1, 320]               0
#           Linear-236                 [-1, 1000]         321,000
#          Softmax-237                 [-1, 1000]               0
# ================================================================
# Total params: 3,851,454
# Trainable params: 3,851,454
# Non-trainable params: 0
# ---------------------------------- end ------------------------------------ #