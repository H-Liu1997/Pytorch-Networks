# --------------------------------------------------------------------------- #
# ResNet, CVPR2016 bestpaper, https://arxiv.org/abs/1512.03385
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNet18','ResNet34','ResNet50','ResNet101','ResNet152']


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
    def __init__(self,in_dim,out_dim,stride=1):
        super(BottleNeck,self).__init__()
        self.subconv_1 = nn.Sequential(
            nn.Conv2d(in_dim,int(out_dim/self.expansion),1,stride,0,bias=False),
            nn.BatchNorm2d(int(out_dim/self.expansion)),
            nn.ReLU(inplace=True),)
        self.subconv_2 = nn.Sequential(
            nn.Conv2d(int(out_dim/self.expansion),
                      int(out_dim/self.expansion),3,1,1,bias=False),
            nn.BatchNorm2d(int(out_dim/self.expansion)),
            nn.ReLU(inplace=True),)
        self.subconv_3 = nn.Sequential(
            nn.Conv2d(int(out_dim/self.expansion),out_dim,1,1,0,bias=False),
            nn.BatchNorm2d(out_dim),)
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
        x_2 = self.subconv_3(x_1)
        if self.downsample is not None:
            x_input = self.downsample(input_)
            print(x_input.shape)
        x_final = F.relu(x_input+x_2,inplace=True)
        return x_final
    

class ResNet(nn.Module):
    def __init__(self, cfg, logger):
        '''
        block, BLOCK_LIST, in_dim, 
        class_num, BASE=64, use_fc=True, CONV1=(7,2,3),
        MAX_POOL=True, pretrained=False
        '''
        super(ResNet,self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(cfg.IN_DIM,cfg.BASE,cfg.CONV1[0],cfg.CONV1[1],cfg.CONV1[2],bias=False),
            nn.BatchNorm2d(cfg.BASE),
            nn.ReLU(inplace=True),)
        if cfg.MAX_POOL:
            self.maxpool_1 = nn.MaxPool2d(3,2,1)
        else:
            self.maxpool_1 = None
        block = BottleNeck if cfg.BLOCK == 'bottleneck' else BasicBlock
        b_ = block.expansion
        self.layer_1 = self._make_layer(block,cfg.BASE,cfg.BASE*b_,cfg.BLOCK_LIST[0],1)
        self.layer_2 = self._make_layer(block,cfg.BASE*b_,cfg.BASE*2*b_,cfg.BLOCK_LIST[1],2)
        self.layer_3 = self._make_layer(block,cfg.BASE*2*b_,cfg.BASE*4*b_,cfg.BLOCK_LIST[2],2)
        self.layer_4 = self._make_layer(block,cfg.BASE*4*b_,cfg.BASE*8*b_,cfg.BLOCK_LIST[3],2)

        final_feature = cfg.BASE*4*b_ if cfg.BLOCK_LIST[3] == 0 else cfg.BASE*8*b_
        if cfg.USE_FC:
            self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
            self.fc_1 = nn.Sequential(
                nn.Flatten(),
                nn.Linear(final_feature,cfg.CLASS_NUM),
                nn.Softmax(dim = 1),)
        else:
            self.avgpool_1 = None
        self.logger = logger
        self.pretrained = cfg.PRETRAINED
        self._initialization()
    
    def _initialization(self):
        if self.pretrained is not False:
            self.modules.load_state_dict(model_zoo.load_url(model_urls[self.pretrained]))
            #TODO(liu):check it correct or not.
        else:
            for name, sub_module in self.named_modules():
                if isinstance(sub_module, nn.Conv2d) or isinstance(sub_module, nn.ConvTranspose2d):
                    nn.init.kaiming_normal_(sub_module.weight,mode='fan_out'
                                            ,nonlinearity='relu')
                    if self.logger is not None:
                        self.logger.info('init {}.weight as kaiming_normal_'.format(name))
                    if sub_module.bias is not None:
                        nn.init.constant_(sub_module.bias, 0.0)
                        if self.logger is not None:
                            self.logger.info('init {}.bias as 0'.format(name))
                elif isinstance(sub_module, nn.BatchNorm2d):
                    nn.init.constant_(sub_module.weight,1)
                    nn.init.constant_(sub_module.bias,0)
                    if self.logger is not None:
                        self.logger.info('init {}.weight as constant_ 1'.format(name))
                        self.logger.info('init {}.bias as constant_ 0'.format(name))
            
    def _make_layer(self,block,in_dim,out_dim,layer_num,stride):
        net_layers = []
        if layer_num == 0:
            return None
        else:    
            for layer in range(layer_num):
                if layer == 0:
                    net_layers.append(block(in_dim,out_dim,stride))
                else:
                    net_layers.append(block(out_dim,out_dim,1))
            return nn.Sequential(*net_layers)
                    
    def forward(self,input_):
        x = self.head_conv(input_)
        if self.maxpool_1 is not None:
            x = self.maxpool_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        if self.layer_4 is not None:
            x = self.layer_4(x)
        if self.avgpool_1 is not None:
            x = self.avgpool_1(x)
            x = self.fc_1(x)
        return x       

model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
}

def ResNet18(pretrained = False):
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def ResNet34(pretrained = False):
    model = ResNet(BasicBlock, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet34']))
    return model

def ResNet50(pretrained = False):
    model = ResNet(BottleNeck, [3, 4, 6, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet50']))
    return model

def ResNet101(pretrained = False):
    model = ResNet(BottleNeck, [3, 4, 23, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet101']))
    return model

def ResNet152(pretrained = False):
    model = ResNet(BottleNeck, [3, 8, 36, 3])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet152']))
    return model



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