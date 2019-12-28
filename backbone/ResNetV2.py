# --------------------------------------------------------------------------- #
# ResNetv2, ECCV2016, https://arxiv.org/abs/1603.05027
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
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim,out_dim,3,stride,1,bias=False),)
        self.subconv_2 = nn.Sequential(
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_dim,out_dim,3,1,1,bias=False),)
        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim,out_dim,1,stride,0,bias=False),)
 
    def forward(self,input_):
        x_input = input_
        x_0 = self.subconv_1(input_)
        x_1 = self.subconv_2(x_0)
        if self.downsample is not None:
            x_input = self.downsample(input_) 
        x_final = x_input + x_1
        return x_final


class BottleNeck(nn.Module):
    expansion = 4
    def __init__(self,in_dim,out_dim,stride=1):
        super(BottleNeck,self).__init__()
        self.subconv_1 = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_dim,int(out_dim/self.expansion),1,stride,0,bias=False),)
        self.subconv_2 = nn.Sequential(
            nn.BatchNorm2d(int(out_dim/self.expansion),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_dim/self.expansion),
                      int(out_dim/self.expansion),3,1,1,bias=False),),)
        self.subconv_3 = nn.Sequential(
            nn.BatchNorm2d(int(out_dim/self.expansion)),
            nn.ReLU(inplace=True),
            nn.Conv2d(int(out_dim/self.expansion),out_dim,1,1,0,bias=False),)
        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim,out_dim,1,stride,0,bias=False),
            )

    def forward(self,input_):
        x_input = input_
        x_0 = self.subconv_1(input_)
        x_1 = self.subconv_2(x_0)
        x_2 = self.subconv_3(x_1)
        if self.downsample is not None:
            x_input = self.downsample(input_)
            print(x_input.shape)
        x_final = x_input+x_2
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


# ---------------------------------- notes ---------------------------------- #
# main idea: test best order
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------- resnet18_v2 model summary ----------------------- #
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#        BatchNorm2d-2         [-1, 64, 112, 112]             128
#               ReLU-3         [-1, 64, 112, 112]               0
#          MaxPool2d-4           [-1, 64, 56, 56]               0
#        BatchNorm2d-5           [-1, 64, 56, 56]             128
#                               ...
#              ReLU-64            [-1, 512, 7, 7]               0
#            Conv2d-65            [-1, 512, 7, 7]       2,359,296
#       BatchNorm2d-66            [-1, 512, 7, 7]           1,024
#              ReLU-67            [-1, 512, 7, 7]               0
#            Conv2d-68            [-1, 512, 7, 7]       2,359,296
#        BasicBlock-69            [-1, 512, 7, 7]               0
# AdaptiveAvgPool2d-70            [-1, 512, 1, 1]               0
#           Flatten-71                  [-1, 512]               0
#            Linear-72                 [-1, 1000]         513,000
#           Softmax-73                 [-1, 1000]               0
# ================================================================
# Total params: 11,687,720
# Trainable params: 11,687,720
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 69.50
# Params size (MB): 44.59
# Estimated Total Size (MB): 114.66
# ---------------------------------- end ------------------------------------ #