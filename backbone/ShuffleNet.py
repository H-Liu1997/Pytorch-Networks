# --------------------------------------------------------------------------- #
# ResNet, CVPR2016 bestpaper, https://arxiv.org/abs/1512.03385
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ShuffleNet_V1']


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_dim,out_dim,stride=1):
        super(BasicBlock,self).__init__()
        self.subconv_1 = nn.Sequential(
            nn.Conv2d(in_dim,out_dim,3,stride,1,bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),)
        self.subconv_2 = nn.Sequential(
            nn.Conv2d(out_dim,out_dim,3,1,1,bias=False),
            nn.BatchNorm2d(out_dim))
        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim,out_dim,1,stride,0,bias=False),
                nn.BatchNorm2d(out_dim),
            )
 
    def forward(self,input_):
        x_input = input_
        x_0 = self.subconv_1(input_)
        x_1 = self.subconv_2(x_0)
        if self.downsample is not None:
            x_input = self.downsample(input_) 
        x_final = F.relu(x_input + x_1,inplace=True)
        return x_final


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_dim,out_dim,stride,groups):
        super(BottleNeck,self).__init__()
        self.subconv_1 = nn.Sequential(
            nn.Conv2d(in_dim,int(out_dim/self.expansion),1,stride,0,bias=False),
            nn.BatchNorm2d(int(out_dim/self.expansion)),
            nn.ReLU(inplace=True),)
        self.groups = groups
        self.subconv_2 = nn.Sequential(
            nn.Conv2d(int(out_dim/self.expansion),
                      int(out_dim/self.expansion),3,1,1,bias=False,groups=groups),
            nn.BatchNorm2d(int(out_dim/self.expansion)),
            nn.ReLU(inplace=True),)
        self.subconv_3 = nn.Sequential(
            nn.Conv2d(int(out_dim/self.expansion),out_dim,1,1,0,bias=False),
            nn.BatchNorm2d(out_dim),)
        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else :
            self.downsample = nn.Sequential(
                 
                 nn.AvgPool2d(3,stride,1),
                 nn.Conv2d(in_dim,out_dim,1,1,0,bias=False),
                 nn.BatchNorm2d(out_dim),
            )
    
    def channel_shuffle(self,x):
        N, C, H, W = x.size()
        assert C % self.groups == 0, 'channel num error'
        
        return x.view(N,self.groups,C//self.groups,
                H,W).permute(0, 2, 1, 3, 4).contiguous().view(N, C, H, W)

    def forward(self,input_):
        x_input = input_
        x_0 = self.subconv_1(input_)
        x_0 = self.channel_shuffle(x_0)
        x_1 = self.subconv_2(x_0)
        x_2 = self.subconv_3(x_1)
        if self.downsample is not None:
            x_input = self.downsample(input_)
            print(x_input.shape)
        x_final = F.relu(x_input+x_2,inplace=True)
        return x_final
    

class _ShuffleNet(nn.Module):
    def __init__(self,block_config,groups):
        super(_ShuffleNet,self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(3,24,3,2,1,bias=False),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True),)
        self.maxpool_1 = nn.MaxPool2d(3,2,1)
        self.layer_1 = self._make_layer(24,block_config[0][1],block_config[0][0],groups)
        self.layer_2 = self._make_layer(block_config[0][1],block_config[1][1],block_config[1][0],groups)
        self.layer_3 = self._make_layer(block_config[1][1],block_config[2][1],block_config[2][0],groups)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1536,1000),
            nn.Softmax(dim = 1),)
    
    def _make_layer(self,in_dim,out_dim,layer_num,groups):
        net_layers = []
        net_layers.append(BottleNeck(in_dim,out_dim,2,groups))
        for layer in range(layer_num):
            net_layers.append(BottleNeck(out_dim,out_dim,1,groups))
        return nn.Sequential(*net_layers)
                    
    def forward(self,input_):
        x = self.head_conv(input_)
        x = self.maxpool_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.avgpool_1(x)
        x = self.fc_1(x)
        return x       


def ShuffleNet_V1(pretrained = False):
    model = _ShuffleNet([[3,384],[7,768],[3,1536]],8)
    if pretrained:
        pass
    return model


def _test():
    from torchsummary import summary
    model = ShuffleNet_V1()
    model = model.cuda()
    summary(model,input_size=(3,224,224))

if __name__ == "__main__":
    _test()

# ---------------------------------- notes ---------------------------------- #
# TODO: Check details in downsample ？
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------- shufflenet v1 model summary ---------------------- #
#      BatchNorm2d-153            [-1, 384, 7, 7]             768
#             ReLU-154            [-1, 384, 7, 7]               0
#           Conv2d-155           [-1, 1536, 7, 7]         589,824
#      BatchNorm2d-156           [-1, 1536, 7, 7]           3,072
#       BottleNeck-157           [-1, 1536, 7, 7]               0
# AdaptiveAvgPool2d-158           [-1, 1536, 1, 1]               0
#          Flatten-159                 [-1, 1536]               0
#           Linear-160                 [-1, 1000]       1,537,000
#          Softmax-161                 [-1, 1000]               0
# ================================================================
# Total params: 11,074,720
# Trainable params: 11,074,720
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 109.57
# Params size (MB): 42.25
# Estimated Total Size (MB): 152.39
# ---------------------------------- end ------------------------------------ #