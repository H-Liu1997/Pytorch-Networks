# --------------------------------------------------------------------------- #
# DenseNet, CVPR2017 bestpaper, https://arxiv.org/abs/1608.06993
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['DenseNet121','DenseNet169','DenseNet201','DenseNet264']


class DenseLayer(nn.Module):

    def __init__(self,in_dim,bn_size,k,drop_rate,memory_efficient=False):
        super(DenseLayer,self).__init__()
        self.sublayer = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim,bn_size*k,1,1,0,bias=False),
            nn.BatchNorm2d(bn_size*k),
            nn.ReLU(inplace=True),
            nn.Conv2d(bn_size*k,k,3,1,1,bias=False),)
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient 
        
    def forward(self,input_):
        x = self.sublayer(input_)
        if self.drop_rate > 0:
            x = F.dropout(x, p=self.drop_rate,training=self.training)
        x_final = torch.cat([input_ , x],dim=1)
        return x_final

class DenseBlock(nn.Module):
    ''' official pytorch use nn.ModuleDict, which has self.items'''
    def __init__(self,in_dim,bn_size,layer_num,k,drop_rate, memory_efficient):
        super(DenseBlock,self).__init__()
        self.subblock = self._make_layers(in_dim,layer_num,bn_size,k,
                                          drop_rate, memory_efficient)
    
    def _make_layers(self,in_dim,layer_num,bn_size,k,drop_rate, memory_efficient):
        model_layers = []
        for layer in range(layer_num):
            model_layers.append(DenseLayer(in_dim+layer*k,
            bn_size,k,drop_rate,memory_efficient)
            )
        return nn.Sequential(*model_layers)

    def forward(self,input_):
        x = input_
        for layer in self.subblock:
            x = layer(x)
        return x


class Transition(nn.Sequential):
    ''' nn.Sequential no need overload forward '''
    def __init__(self,in_dim,out_dim):
        super(Transition,self).__init__()
        self.norm = nn.BatchNorm2d(in_dim)
        self.relu = nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(in_dim,out_dim,1,1,0,bias=False)
        self.pool = nn.AvgPool2d(2,2)
        
   
class DenseNet(nn.Module):
    def __init__(self,k,block_list,num_init_features=64, bn_size=4, 
                 drop_rate=0, memory_efficient=False):
        super(DenseNet,self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(3,num_init_features,7,2,3,bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.ReLU(inplace=True),)
        self.maxpool_1 = nn.MaxPool2d(3,2,1)
        self.dense_body, self.final_channels = self._make_layers(num_init_features,
                                  bn_size,block_list,k,drop_rate, memory_efficient)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.final_channels,1000),
            nn.Softmax(dim = 1),)
        self._initialization()
    
    def _initialization(self):
        for m in self.modules():
            if isinstance (m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def _make_layers(self,num_init_features,bn_size,block_list,k, 
                     drop_rate, memory_efficient):
        net_layers = []
        for layer_num in block_list:
            net_layers.append(DenseBlock(num_init_features,bn_size,layer_num,
                                         k, drop_rate, memory_efficient))
            net_layers.append(Transition(num_init_features+layer_num*k,
                                       (num_init_features+layer_num*k)//2))
            num_init_features = (num_init_features+layer_num*k)//2                           
        return nn.Sequential(*net_layers),num_init_features
                    
    def forward(self,input_):
        x = self.head_conv(input_)
        x = self.maxpool_1(x)
        x = self.dense_body(x)
        x = self.avgpool_1(x)
        x = self.fc_1(x)
        return x       

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}

def DenseNet121(pretrained = False):
    model = DenseNet(32, [6, 12, 24, 16])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet121']))
    return model

def DenseNet169(pretrained = False):
    model = DenseNet(32, [6, 12, 32, 32])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet169']))
    return model

def DenseNet201(pretrained = False):
    model = DenseNet(32, [6, 12, 48, 32])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['densenet201']))
    return model

def DenseNet264(pretrained = False):
    model = DenseNet(32, [6, 12, 64, 48])
    return model


def _test():
    from torchsummary import summary
    model = DenseNet264()
    model = model.cuda()
    summary(model,input_size=(3,224,224))

if __name__ == "__main__":
    _test()


# ------------------------------- mistakes ---------------------------------- #
# nn.sequential no need overload forward
# if name is needed, use nn.ModuleDict
# ---------------------------------- end ------------------------------------ #


# ------------------------------- background -------------------------------- #
#
# ---------------------------------- end ------------------------------------ #


# ---------------------------------- notes ---------------------------------- #
# main idea: Dense short cut in channel level, not elements level
# parameters: 35M dense264, 7M dense121, BN+ReLU
# sgd+momentum+Nesterov 1e-1 0.9 divide 10 * 2 
# batch size 256
# weight decay 1e-4
# input: resize and crop samll side to 256×256 then augment to 224
# output: linear 1000 + softmax
# TODO: Check details in small dense net
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------ dense121 model summary --------------------------- #
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#        BatchNorm2d-5           [-1, 64, 56, 56]             128
#               ReLU-6           [-1, 64, 56, 56]               0
#             Conv2d-7          [-1, 128, 56, 56]           8,192
#        BatchNorm2d-8          [-1, 128, 56, 56]             256
#               ReLU-9          [-1, 128, 56, 56]               0
#            Conv2d-10           [-1, 32, 56, 56]          36,864
#                                  ...
#             ReLU-428           [-1, 1024, 7, 7]               0
#           Conv2d-429            [-1, 512, 7, 7]         524,288
#        AvgPool2d-430            [-1, 512, 3, 3]               0
# AdaptiveAvgPool2d-431            [-1, 512, 1, 1]               0
#          Flatten-432                  [-1, 512]               0
#           Linear-433                 [-1, 1000]         513,000
#          Softmax-434                 [-1, 1000]               0
# ================================================================
# Total params: 7,991,144
# Trainable params: 7,991,144
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 383.54
# Params size (MB): 30.48
# Estimated Total Size (MB): 414.60
# ---------------------------------- end ------------------------------------ #
