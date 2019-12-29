# --------------------------------------------------------------------------- #
# SEmodule, CVPR2018, https://arxiv.org/abs/1709.01507
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
from torch import nn
from torch.nn import functional as F


class Swish(nn.Module):
    def __init__(self):
        super(Swish,self).__init__()
    
    def forward(self,input_):
        return input_*F.relu(input_,inplace=True)


class _SElayer(nn.Module):
    def __init__(self,in_dim,ratio):
        super(_SElayer,self).__init__()
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        reduced_dim = max(1, in_dim//ratio)
        self.fc1 = nn.Sequential(nn.Flatten(),
                   nn.Linear(in_dim, reduced_dim),
                   Swish(),
                   nn.Linear(reduced_dim, in_dim),
                   nn.Softmax(dim=1),)

    def forward(self,input_):
        x_input = input_
        x = self.gap(input_)
        x = self.fc1(x)
        x = x.view(-1,x_input.shape[1],1,1)
        x = x_input * x.expand_as(x_input) 
        return x


class _DWConv(nn.Sequential):
    def __init__(self,in_dim,k_size,stride):
        super(_DWConv,self).__init__()
        self.add_module('depthwise',nn.Conv2d(in_dim,in_dim,k_size,stride,
                       (k_size-1)//2,bias=False,groups=in_dim)),
        self.add_module('bn',nn.BatchNorm2d(in_dim)),
        self.add_module('relu',Swish(),)
        

class _Conv(nn.Sequential):
    def __init__(self,in_dim,out_dim,k,s,p):
        super(_Conv,self).__init__()
        self.add_module('conv',nn.Conv2d(in_dim,out_dim,k,s,p,bias=False)),
        self.add_module('bn',nn.BatchNorm2d(out_dim)),
        self.add_module('relu',Swish())

class _MBConv(nn.Module):
    def __init__(self,in_dim,out_dim,depth,k,stride,drop_connect_rate,SE_ratio=None):
        super(_MBConv,self).__init__()
        layers = []
        if in_dim != depth*in_dim:
            layers.append(_Conv(in_dim,depth*in_dim,1,1,0))
        layers.append(_DWConv(depth*in_dim,k,stride))
        if stride != 1 or in_dim != out_dim:
            # self.Downsample =  nn.Sequential(
            # nn.Conv2d(in_dim,out_dim,1,stride,0,bias=False),
            # nn.BatchNorm2d(out_dim)
            self.use_res = False

        else:
            #self.Downsample = None
            self.use_res = True
        if SE_ratio is not None:
            layers.append(_SElayer(depth*in_dim,SE_ratio))
        layers.append( nn.Sequential(
            nn.Conv2d(depth*in_dim,out_dim,1,1,0,bias=False),
            nn.BatchNorm2d(out_dim)
        ))
        self.conv_total = nn.Sequential(*layers)
        self.drop_connect_rate = drop_connect_rate
        self.Swish = Swish()

    def _drop_connect(self, x):
        if not self.training:
            return x
        keep_prob = 1.0 - self.drop_connect_rate
        batch_size = x.size(0)
        random_tensor = keep_prob
        random_tensor += torch.rand(batch_size, 1, 1, 1, device=x.device)
        binary_tensor = random_tensor.floor()
        return x.div(keep_prob) * binary_tensor
        
    def forward(self,input_):
        x = self.conv_total(input_)
        x = self._drop_connect(x)
        if self.use_res:
            x += input_
        x = self.Swish(x)
        return x


class EfficientNet_B0(nn.Sequential):
    def __init__(self):
        super(EfficientNet_B0,self).__init__()
        self.HeadConv = _Conv(3,32,3,2,1)
        self.MBConv1_1 = _MBConv(32,16,1,3,1,0.2,4)

        self.MBConv6_1 = _MBConv(16,24,6,3,2,0.2,4)
        self.MBConv6_2 = _MBConv(24,24,6,3,1,0.2,4)

        self.MBConv6_3 = _MBConv(24,40,6,5,2,0.2,4)
        self.MBConv6_4 = _MBConv(40,40,6,5,1,0.2,4)

        self.MBConv6_5 = _MBConv(40,80,6,3,2,0.2,4)
        self.MBConv6_6 = _MBConv(80,80,6,3,1,0.2,4)
        self.MBConv6_7 = _MBConv(80,80,6,3,1,0.2,4)

        self.MBConv6_8 = _MBConv(80,112,6,5,1,0.2,4)
        self.MBConv6_9 = _MBConv(112,112,6,5,1,0.2,4)
        self.MBConv6_10 = _MBConv(112,112,6,5,1,0.2,4)

        self.MBConv6_11 = _MBConv(112,192,6,5,2,0.2,4)
        self.MBConv6_12 = _MBConv(192,192,6,5,1,0.2,4)
        self.MBConv6_13 = _MBConv(192,192,6,5,1,0.2,4)
        self.MBConv6_14 = _MBConv(192,192,6,5,1,0.2,4)

        self.MBConv6_15 = _MBConv(192,320,6,3,1,0.2,4)

        self.logits = nn.Sequential(
            #_Conv(320,1280,1,1,0),
            nn.AdaptiveAvgPool2d((1,1)),
            nn.Flatten(),
            nn.Linear(320,1000),
            nn.Softmax(dim=1))


def _test():
    from torchsummary import summary
    model = EfficientNet_B0()
    torch.cuda.set_device(1)
    model = model.cuda()
    summary(model,input_size=(3,224,224))

if __name__ == "__main__":
    _test()


# ---------------------------------- notes ---------------------------------- #
# main idea: new nas method
# TODO: Training check: False
# TODO: check parameters
# ---------------------------------- end ------------------------------------ #


# ------------------------- Efficientb0 model summary ------------------------ #
#           Linear-266                 [-1, 1152]         332,928
#          Softmax-267                 [-1, 1152]               0
#         _SElayer-268           [-1, 1152, 7, 7]               0
#           Conv2d-269            [-1, 320, 7, 7]         368,640
#      BatchNorm2d-270            [-1, 320, 7, 7]             640
#             ReLU-271            [-1, 320, 7, 7]               0
#          _MBConv-272            [-1, 320, 7, 7]               0
# AdaptiveAvgPool2d-273            [-1, 320, 1, 1]               0
#          Flatten-274                  [-1, 320]               0
#           Linear-275                 [-1, 1000]         321,000
#          Softmax-276                 [-1, 1000]               0
# ================================================================
# Total params: 7,051,688
# Trainable params: 7,051,688
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 175.13
# Params size (MB): 26.90
# Estimated Total Size (MB): 202.60-
# ---------------------------------- end ------------------------------------ #