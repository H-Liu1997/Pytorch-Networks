# --------------------------------------------------------------------------- #
# ResNet, CVPR2016 bestpaper, https://arxiv.org/abs/1512.03385
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


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
#        BatchNorm2d-9           [-1, 64, 56, 56]             128
#        BasicBlock-10           [-1, 64, 56, 56]               0
#            Conv2d-11           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-12           [-1, 64, 56, 56]             128
#              ReLU-13           [-1, 64, 56, 56]               0
#            Conv2d-14           [-1, 64, 56, 56]          36,864
#       BatchNorm2d-15           [-1, 64, 56, 56]             128
#        BasicBlock-16           [-1, 64, 56, 56]               0
#            Conv2d-17          [-1, 128, 28, 28]          73,728
#       BatchNorm2d-18          [-1, 128, 28, 28]             256
#              ReLU-19          [-1, 128, 28, 28]               0
#            Conv2d-20          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-21          [-1, 128, 28, 28]             256
#            Conv2d-22          [-1, 128, 28, 28]           8,192
#       BatchNorm2d-23          [-1, 128, 28, 28]             256
#        BasicBlock-24          [-1, 128, 28, 28]               0
#            Conv2d-25          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-26          [-1, 128, 28, 28]             256
#              ReLU-27          [-1, 128, 28, 28]               0
#            Conv2d-28          [-1, 128, 28, 28]         147,456
#       BatchNorm2d-29          [-1, 128, 28, 28]             256
#        BasicBlock-30          [-1, 128, 28, 28]               0
#            Conv2d-31          [-1, 256, 14, 14]         294,912
#       BatchNorm2d-32          [-1, 256, 14, 14]             512
#              ReLU-33          [-1, 256, 14, 14]               0
#            Conv2d-34          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-35          [-1, 256, 14, 14]             512
#            Conv2d-36          [-1, 256, 14, 14]          32,768
#       BatchNorm2d-37          [-1, 256, 14, 14]             512
#        BasicBlock-38          [-1, 256, 14, 14]               0
#            Conv2d-39          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-40          [-1, 256, 14, 14]             512
#              ReLU-41          [-1, 256, 14, 14]               0
#            Conv2d-42          [-1, 256, 14, 14]         589,824
#       BatchNorm2d-43          [-1, 256, 14, 14]             512
#        BasicBlock-44          [-1, 256, 14, 14]               0
#            Conv2d-45            [-1, 512, 7, 7]       1,179,648
#       BatchNorm2d-46            [-1, 512, 7, 7]           1,024
#              ReLU-47            [-1, 512, 7, 7]               0
#            Conv2d-48            [-1, 512, 7, 7]       2,359,296
#       BatchNorm2d-49            [-1, 512, 7, 7]           1,024
#            Conv2d-50            [-1, 512, 7, 7]         131,072
#       BatchNorm2d-51            [-1, 512, 7, 7]           1,024
#        BasicBlock-52            [-1, 512, 7, 7]               0
#            Conv2d-53            [-1, 512, 7, 7]       2,359,296
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
    def __init__(self,block,block_list):
        super(ResNet,self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)
        self.maxpool_1 = nn.MaxPool2d(3,2,1)
        b_ = block.expansion
        self.layer_1 = self._make_layer(block,64,64*b_,block_list[0],1)
        self.layer_2 = self._make_layer(block,64*b_,128*b_,block_list[1],2)
        self.layer_3 = self._make_layer(block,128*b_,256*b_,block_list[2],2)
        self.layer_4 = self._make_layer(block,256*b_,512*b_,block_list[3],2)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512*b_,1000),
            nn.Softmax(dim = 1),)
        self._initialization()
    
    def _initialization(self):
        for m in self.modules():
            if isinstance (m,nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,mode='fan_out'
                                        ,nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight,1)
                nn.init.constant_(m.bias,0)
    
    def _make_layer(self,block,in_dim,out_dim,layer_num,stride):
        net_layers = []
        for layer in range(layer_num):
            if layer == 0:
                net_layers.append(block(in_dim,out_dim,stride))
            else:
                net_layers.append(block(out_dim,out_dim,1))
        return nn.Sequential(*net_layers)
                    
    def forward(self,input_):
        x = self.head_conv(input_)
        x = self.maxpool_1(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
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


def _test():
    from torchsummary import summary
    model = ResNet18()
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