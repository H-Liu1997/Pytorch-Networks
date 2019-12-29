# --------------------------------------------------------------------------- #
# SEmodule, CVPR2018, https://arxiv.org/abs/1709.01507
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SElayer']



class SElayer(nn.Module):
    def __init__(self,in_dim,ratio):
        super(SElayer,self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        reduced_dim = max(1, in_dim//ratio)
        self.fc1 = nn.Sequential(nn.Flatten(),
                   nn.Linear(in_dim, reduced_dim),
                   #_Swish(),
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

class TestNet(nn.Sequential):
    def __init__(self):
        super(TestNet,self).__init__()
        self.conv1 = nn.Conv2d(3,64,3,1,1)
        self.se = SElayer(64,16)

def _test():
    from torchsummary import summary
    model = TestNet()
    torch.cuda.set_device(1)
    model = model.cuda()
    summary(model,input_size=(3,224,224))

if __name__ == "__main__":
    _test()

# ------------------------------- mistakes ---------------------------------- #
# downsample also need add batchnorm
# add first, then relu
# add input, not first conv output.
# no bias for all conv layers
# when using /, need add int()
# usually we use fin_in for LeCun and he init, here we use fan_out
# ---------------------------------- end ------------------------------------ #


# ---------------------------------- notes ---------------------------------- #
# main idea: short cut connection
# parameters: 2.5M Res50, 6M Res152, 1.1M Res20, BN+ReLU
# sgd+momentum 1e-1 0.9 divide 10 * 3 
# batch size 256
# weight decay 1e-4
# input: resize and crop samll side to 256×256 then augment to 224
# output: linear 1000 + softmax
# TODO: Check details in training,testing. bn-relu-conv?
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------- resnet18 model summary -------------------------- #
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#             Conv2d-5           [-1, 64, 56, 56]          36,864
#        BatchNorm2d-6           [-1, 64, 56, 56]             128
#               ReLU-7           [-1, 64, 56, 56]               0
#             Conv2d-8           [-1, 64, 56, 56]          36,864
#                                 ...
#       BatchNorm2d-54            [-1, 512, 7, 7]           1,024
#              ReLU-55            [-1, 512, 7, 7]               0
#            Conv2d-56            [-1, 512, 7, 7]       2,359,296
#       BatchNorm2d-57            [-1, 512, 7, 7]           1,024
#        BasicBlock-58            [-1, 512, 7, 7]               0
# AdaptiveAvgPool2d-59            [-1, 512, 1, 1]               0
#           Flatten-60                  [-1, 512]               0
#            Linear-61                 [-1, 1000]         513,000
#           Softmax-62                 [-1, 1000]               0
# ================================================================
# Total params: 11,689,512
# Trainable params: 11,689,512
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 57.06
# Params size (MB): 44.59
# Estimated Total Size (MB): 102.23
# ---------------------------------- end ------------------------------------ #