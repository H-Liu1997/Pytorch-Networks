# --------------------------------------------------------------------------- #
# ResNeXt, CVPR2017, https://arxiv.org/abs/1611.05431
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['ResNeXt50']


class BasicBlock(nn.Module):
    expansion = 2
    def __init__(self,in_dim,out_dim,stride=1,cardinality=32):
        super(BasicBlock,self).__init__()
        self.layers_ = self._make_layers(in_dim,out_dim,stride,cardinality)
        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim,out_dim,1,stride,0,bias=False),
                nn.BatchNorm2d(out_dim),
            )

    def _make_layers(self,in_dim,out_dim,stride,cardinality):
        layers = []
        for group in range(cardinality):
            layers.append(nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim,out_dim//self.expansion//cardinality,1,stride,0,bias=False),
                nn.BatchNorm2d(out_dim//self.expansion//cardinality),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim//self.expansion//cardinality,
                out_dim//self.expansion//cardinality,3,1,1,bias=False),
                nn.BatchNorm2d(out_dim//self.expansion//cardinality),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim//self.expansion//cardinality,out_dim,1,1,0,bias=False),
            ))
        return nn.Sequential(*layers)

    def forward(self,input_):
        x_input = input_
        if self.downsample is not None:
            x_input = self.downsample(input_) 
        x_sum = torch.zeros_like(x_input)
        for group in self.layers_:
            x_ = group(input_)
            x_sum += x_   
        x_final = x_input + x_sum
        return x_final


class BasicBlock_pytorch_group(nn.Module):
    expansion = 2
    def __init__(self,in_dim,out_dim,stride=1,cardinality=32):
        super(BasicBlock_pytorch_group,self).__init__()
        self.layers_ = nn.Sequential(
                nn.BatchNorm2d(in_dim),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_dim,out_dim//self.expansion,1,stride,0,bias=False),
                nn.BatchNorm2d(out_dim//self.expansion),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim//self.expansion,
                out_dim//self.expansion,3,1,1,bias=False,groups = cardinality),
                nn.BatchNorm2d(out_dim//self.expansion),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_dim//self.expansion,out_dim,1,1,0,bias=False),
            )
        if in_dim == out_dim and stride == 1:
            self.downsample = None
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim,out_dim,1,stride,0,bias=False),
                nn.BatchNorm2d(out_dim),
            )

    def forward(self,input_):
        x_input = input_
        if self.downsample is not None:
            x_input = self.downsample(input_) 
        x_ =  self.layers_(input_)
        x_final = x_input + x_
        return x_final


class ResNet(nn.Module):
    def __init__(self,block,block_list,cardinality):
        super(ResNet,self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),)
        self.maxpool_1 = nn.MaxPool2d(3,2,1)
        b_ = block.expansion
        self.layer_1 = self._make_layer(block,64,128*b_,block_list[0],1,cardinality)
        self.layer_2 = self._make_layer(block,128*b_,256*b_,block_list[1],2,cardinality)
        self.layer_3 = self._make_layer(block,256*b_,512*b_,block_list[2],2,cardinality)
        self.layer_4 = self._make_layer(block,512*b_,1024*b_,block_list[3],2,cardinality)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*b_,1000),
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
    
    def _make_layer(self,block,in_dim,out_dim,layer_num,stride,cardinality):
        net_layers = []
        for layer in range(layer_num):
            if layer == 0:
                net_layers.append(block(in_dim,out_dim,stride,cardinality))
            else:
                net_layers.append(block(out_dim,out_dim,1,cardinality))
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


def ResNeXt50(pretrained = False):
    model = ResNet(BasicBlock_pytorch_group, [3, 4, 6, 3],32)
    return model

    
def _test():
    from torchsummary import summary
    model = ResNeXt50()
    model = model.cuda()
    summary(model,input_size=(3,224,224))

if __name__ == "__main__":
    _test()


# ---------------------------------- notes ---------------------------------- #
# TODO: self implement 10*larger than pytorch.
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------- resnext50 model summary ------------------------- #
# AdaptiveAvgPool2d-173           [-1, 2048, 1, 1]               0
#          Flatten-174                 [-1, 2048]               0
#           Linear-175                 [-1, 1000]       2,049,000
#          Softmax-176                 [-1, 1000]               0
# ================================================================
# Total params: 25,024,936
# Trainable params: 25,024,936
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 339.22
# Params size (MB): 95.46
# Estimated Total Size (MB): 435.26
# ----------------------------------------------------------------
# ---------------------------------- end ------------------------------------ #