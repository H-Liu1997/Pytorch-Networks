# --------------------------------------------------------------------------- #
# Inception v1, CVPR2015, https://arxiv.org/abs/1409.4842
# Inception v2+v3 CVPR2016, https://arxiv.org/abs/1512.00567
# Inception v4+res CVPR2017, https://arxiv.org/abs/1602.07261
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['Inception_v1','Inception_v3','Inception_v4','Inception_res']


class _InceptionV1Block(nn.Module):

    def __init__(self,in_dim,conv1,conv3_r,conv3,conv5_r,conv5,pool):
        super(_InceptionV1Block,self).__init__()
        self.conv1_branch = nn.Sequential(
                            nn.Conv2d(in_dim,conv1,1,1,0),
                            nn.ReLU(inplace=True),)                
        self.conv3_branch = nn.Sequential(
                            nn.Conv2d(in_dim,conv3_r,1,1,0),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(conv3_r,conv3,3,1,1),
                            nn.ReLU(inplace=True),) 
        self.conv5_branch = nn.Sequential(
                            nn.Conv2d(in_dim,conv5_r,1,1,0),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(conv5_r,conv5,5,1,2),
                            nn.ReLU(inplace=True),)
        self.pool_branch = nn.Sequential(
                           nn.MaxPool2d(3,1,1),
                           nn.Conv2d(in_dim,pool,1,1,0),
                           nn.ReLU(inplace=True),) 
        
    def forward(self,input_):
        x_1 = self.conv1_branch(input_)
        x_3 = self.conv3_branch(input_)
        x_5 = self.conv5_branch(input_)
        x_p = self.pool_branch(input_)
        x_final = torch.cat([x_1,x_3,x_5,x_p],dim=1)
        return x_final

   
class Inception_v1(nn.Module):
    def __init__(self):
        super(Inception_v1,self).__init__()
        self.head_conv = nn.Sequential(
            nn.Conv2d(3,64,7,2,3,bias=False),
            nn.ReLU(inplace=True),)
        self.maxpool_1 = nn.MaxPool2d(3,2,1)
        #TODO: localRespNorm?
        self.conv2 = nn.Sequential(
            nn.Conv2d(64,192,3,1,1,bias=False),
            nn.ReLU(inplace=True),)
        self.maxpool_2 = nn.MaxPool2d(3,2,1)
        self.inception3a = _InceptionV1Block(192,64, 96, 128,16,32, 32)
        self.inception3b = _InceptionV1Block(256,128,128,192,32,96, 64)
        self.maxpool_3 = nn.MaxPool2d(3,2,1)
        self.inception4a = _InceptionV1Block(480,192,96, 208,16,48, 64)
        self.inception4b = _InceptionV1Block(512,160,112,224,24,64, 64)
        self.inception4c = _InceptionV1Block(512,128,128,256,24,64, 64)
        self.inception4d = _InceptionV1Block(512,112,144,288,32,64, 64)
        self.inception4e = _InceptionV1Block(528,256,160,320,32,128,128)
        self.maxpool_4 = nn.MaxPool2d(3,2,1)
        self.inception5a = _InceptionV1Block(832,256,160,320,32,128,128)
        self.inception5b = _InceptionV1Block(832,384,192,384,48,128,128)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Sequential(#from 4a 4*4*512
            nn.AvgPool2d(5,3,0),
            nn.Conv2d(512,128,1,1,0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024,1000),
            nn.Softmax(dim = 1),)
        self.fc_2 = nn.Sequential(#from 4d 4*4*528
            nn.AvgPool2d(5,3,0),
            nn.Conv2d(528,128,1,1,0),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024,1000),
            nn.Softmax(dim = 1),)
        self.fc_final = nn.Sequential(
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(1024,1000),
            nn.Softmax(dim = 1),)
        self._initialization()
    
    def _initialization(self):
        for m in self.modules():
            if isinstance (m,nn.Conv2d):
                nn.init.normal_(m.weight,mean=0,std=0.01)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)
                    
    def forward(self,input_):
        x = self.head_conv(input_)
        x = self.maxpool_1(x)
        x = self.conv2(x)
        x = self.maxpool_2(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool_3(x)
        x = self.inception4a(x)
        x_loss1 = self.fc_1(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x_loss2 = self.fc_2(x)
        x = self.inception4e(x)
        x = self.maxpool_4(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.avgpool_1(x)
        x = self.fc_final(x)
        return x_loss1,x_loss2,x       

model_urls = {
    'densenet121': 'https://download.pytorch.org/models/densenet121-a639ec97.pth',
    'densenet169': 'https://download.pytorch.org/models/densenet169-b2777c0a.pth',
    'densenet201': 'https://download.pytorch.org/models/densenet201-c1103571.pth',
    'densenet161': 'https://download.pytorch.org/models/densenet161-8d451a50.pth',
}


def _test():
    from torchsummary import summary
    model = Inception_v1()
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
