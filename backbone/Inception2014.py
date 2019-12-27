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
                            nn.Conv2d(in_dim,conv1,1,1,0,bias=False),
                            nn.ReLU(inplace=True),)                
        self.conv3_branch = nn.Sequential(
                            nn.Conv2d(in_dim,conv3_r,1,1,0,bias=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(conv3_r,conv3,3,1,1,bias=False),
                            nn.ReLU(inplace=True),) 
        self.conv5_branch = nn.Sequential(
                            nn.Conv2d(in_dim,conv5_r,1,1,0,bias=False),
                            nn.ReLU(inplace=True),
                            nn.Conv2d(conv5_r,conv5,5,1,2,bias=False),
                            nn.ReLU(inplace=True),)
        self.pool_branch = nn.Sequential(
                           nn.MaxPool2d(3,1,1),
                           nn.Conv2d(in_dim,pool,1,1,0,bias=False),
                           nn.ReLU(inplace=True),) 
        
    def forward(self,input_):
        x_1 = self.conv1_branch(input_)
        x_3 = self.conv3_branch(input_)
        x_5 = self.conv5_branch(input_)
        x_p = self.pool_branch(input_)
        x_final = torch.cat([x_1,x_3,x_5,x_p],dim=1)
        return x_final


class _InceptionV3Block_5(nn.Module):

    def __init__(self,in_dim,conv1,conv3_r,conv3,conv5_r,conv5,pool):
        super(_InceptionV3Block_5,self).__init__()
        self.conv1_branch = BasicConv2d(in_dim,conv1,1,1,0)              
        self.conv3_branch = BasicConv2d(in_dim,conv3_r,1,1,0)                   
        self.conv3_1 = BasicConv2d(conv3_r,conv3,(3,1),1,(1,0))
        self.conv3_2 = BasicConv2d(conv3_r,conv3,(1,3),1,(0,1))
        self.conv5_branch = nn.Sequential(
                            BasicConv2d(in_dim,conv5_r,1,1,0), 
                            BasicConv2d(conv5_r,conv5,3,1,1),)
        self.conv5_1 = BasicConv2d(conv5,conv5,(3,1),1,(1,0))
        self.conv5_2 = BasicConv2d(conv5,conv5,(1,3),1,(0,1))
        self.pool_branch = nn.Sequential(
                           nn.AvgPool2d(3,1,1),
                           BasicConv2d(in_dim,pool,1,1,0),) 
        
    def forward(self,input_):
        x_1 = self.conv1_branch(input_)
        x_3 = self.conv3_branch(input_)
        x_3_1 = self.conv3_1(x_3)
        x_3_2 = self.conv3_2(x_3)
        x_5 = self.conv5_branch(input_)
        x_5_1 = self.conv5_1(x_5)
        x_5_2 = self.conv5_2(x_5)
        x_p = self.pool_branch(input_)
        x_final = torch.cat([x_1,x_3_1,x_3_2,x_5_1,x_5_2,x_p],dim=1)
        return x_final

class _InceptionV3Block_4(nn.Module):
    def __init__(self,in_dim,conv3_r,conv3,conv5,stride):
        super(_InceptionV3Block_4,self).__init__()        
        self.conv3_branch = nn.Sequential(
                            BasicConv2d(in_dim,conv3_r,1,1,0),
                            BasicConv2d(conv3_r,conv3,3,stride,0))
        self.conv5_branch = nn.Sequential(
                            BasicConv2d(in_dim,conv5,1,1,0),
                            BasicConv2d(conv5,conv5,(1,7),1,(0,3)),
                            BasicConv2d(conv5,conv5,(7,1),1,(3,0)),
                            BasicConv2d(conv5,conv5,3,stride,0))
        self.pool_branch = nn.MaxPool2d(3,stride,0)
        
    def forward(self,input_):
        x_3 = self.conv3_branch(input_)
        x_5 = self.conv5_branch(input_)
        x_p = self.pool_branch(input_)
        x_final = torch.cat([x_3,x_5,x_p],dim=1)
        return x_final

class _InceptionV3Block_3(nn.Module):
    '''original paper use 1*7 '''
    def __init__(self,in_dim,conv1,conv3_r,conv3,conv5_r,conv5,pool):
        super(_InceptionV3Block_3,self).__init__()
        self.conv1_branch = BasicConv2d(in_dim,conv1,1,1,0)              
        self.conv3_branch = nn.Sequential(
                            BasicConv2d(in_dim,conv3_r,1,1,0),
                            BasicConv2d(conv3_r,conv3_r,(1,7),1,(0,3)),
                            BasicConv2d(conv3_r,conv3,(7,1),1,(3,0)),)
        self.conv5_branch = nn.Sequential(
                            BasicConv2d(in_dim,conv5_r,1,1,0),
                            BasicConv2d(conv5_r,conv5_r,(7,1),1,(3,0)),
                            BasicConv2d(conv5_r,conv5_r,(1,7),1,(0,3)),
                            BasicConv2d(conv5_r,conv5_r,(7,1),1,(3,0)),
                            BasicConv2d(conv5_r,conv5,(1,7),1,(0,3)),)
        self.pool_branch = nn.Sequential(
                           nn.AvgPool2d(3,1,1),
                           BasicConv2d(in_dim,pool,1,1,0),)
        
    def forward(self,input_):
        x_1 = self.conv1_branch(input_)
        x_3 = self.conv3_branch(input_)
        x_5 = self.conv5_branch(input_)
        x_p = self.pool_branch(input_)
        x_final = torch.cat([x_1,x_3,x_5,x_p],dim=1)
        return x_final

class _InceptionV3Block_1(nn.Module):
    def __init__(self,in_dim,conv1,conv3_r,conv3,conv5_r,conv5,pool):
        super(_InceptionV3Block_1,self).__init__()
        self.conv1_branch = BasicConv2d(in_dim,conv1,1,1,0)               
        self.conv3_branch = nn.Sequential(
                            BasicConv2d(in_dim,conv3_r,1,1,0),
                            BasicConv2d(conv3_r,conv3,5,1,2),) 
        self.conv5_branch = nn.Sequential(
                            BasicConv2d(in_dim,conv5_r,1,1,0),
                            BasicConv2d(conv5_r,conv5,3,1,1),
                            BasicConv2d(conv5,conv5,3,1,1),)
        self.pool_branch = nn.Sequential(
                           nn.AvgPool2d(3,1,1),
                           BasicConv2d(in_dim,pool,1,1,0),) 
        
    def forward(self,input_):
        x_1 = self.conv1_branch(input_)
        x_3 = self.conv3_branch(input_)
        x_5 = self.conv5_branch(input_)
        x_p = self.pool_branch(input_)
        x_final = torch.cat([x_1,x_3,x_5,x_p],dim=1)
        return x_final
   
class _InceptionV3Block_2(nn.Module):
    def __init__(self,in_dim,conv3,conv5_r,conv5,stride):
        super(_InceptionV3Block_2,self).__init__()              
        self.conv3_branch = BasicConv2d(in_dim,conv3,3,stride,0)
        self.conv5_branch = nn.Sequential(
            BasicConv2d(in_dim,conv5_r,1,1,0),
            BasicConv2d(conv5_r,conv5,3,1,1),
            BasicConv2d(conv5,conv5,3,stride,0),)
        self.pool_branch = nn.MaxPool2d(3,stride,0)
    
    def forward(self,input_):
        x_3 = self.conv3_branch(input_)
        x_5 = self.conv5_branch(input_)
        x_p = self.pool_branch(input_)
        x_final = torch.cat([x_3,x_5,x_p],dim=1)
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
            nn.Conv2d(512,128,1,1,0,bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128*16,1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.7),
            nn.Linear(1024,1000),
            nn.Softmax(dim = 1),)
        self.fc_2 = nn.Sequential(#from 4d 4*4*528
            nn.AvgPool2d(5,3,0),
            nn.Conv2d(528,128,1,1,0,bias=False),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128*16,1024),
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
        #TODO:Check it again
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


class BasicConv2d(nn.Module):

    def __init__(self, in_dim, out_dim, k, s, p):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, k,s,p,bias=False,)
        self.bn = nn.BatchNorm2d(out_dim, eps=0.001)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return F.relu(x, inplace=True)

class Inception_v3(nn.Module):
    def __init__(self):
        super(Inception_v3,self).__init__()
        self.head_conv = nn.Sequential(
            BasicConv2d(3,32,3,2,0),
            BasicConv2d(32,32,3,1,0),
            BasicConv2d(32,64,3,1,1), )
        self.maxpool_1 = nn.MaxPool2d(3,2,0)
        # TODO: there are some difference between pytorch code and offical paper
        # origin paper don't have details
        self.conv2 = nn.Sequential(
            BasicConv2d(64,80,1,1,0),
            BasicConv2d(80,192,3,1,0),)
        self.maxpool_2 = nn.MaxPool2d(3,2,0)

        self.inception3a = _InceptionV3Block_1(192,64, 48, 64, 64, 96, 32)
        self.inception3b = _InceptionV3Block_1(256,64, 48, 64, 64, 96, 64)
        self.inception3c = _InceptionV3Block_1(288,64, 48, 64, 64, 96, 64)

        self.inception4a = _InceptionV3Block_2(288,384,64, 96, 2)

        self.inception4b = _InceptionV3Block_3(768,192,128,192,128,192,192)
        self.inception4c = _InceptionV3Block_3(768,192,160,192,160,192,192)
        self.inception4d = _InceptionV3Block_3(768,192,160,192,160,192,192)
        self.inception4e = _InceptionV3Block_3(768,192,192,192,192,192,192)

        self.inception5a = _InceptionV3Block_4(768,192,320,192,2)

        self.inception5b = _InceptionV3Block_5(1280,320,384,384,448,384,192)
        self.inception5c = _InceptionV3Block_5(2048,320,384,384,448,384,192)
        self.avgpool_1 = nn.AdaptiveAvgPool2d((1,1))
        self.fc_1 = nn.Sequential(#from 4a 4*4*512
            nn.AvgPool2d(5,3,0),
            BasicConv2d(768,128,1,1,0),
            BasicConv2d(128,768,5,1,0),
            nn.Flatten(),
            nn.Linear(768,1000),
            nn.Softmax(dim = 1),)
        self.fc_final = nn.Sequential(
            nn.Dropout(0.4),
            nn.Flatten(),
            nn.Linear(2048,1000),
            nn.Softmax(dim = 1),)
        self._initialization()
    
    def _initialization(self):
        #TODO:Check it again
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
        x = self.inception3c(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x_loss1 = self.fc_1(x)
        x = self.inception5a(x)
        x = self.inception5b(x)
        x = self.inception5c(x)

        x = self.avgpool_1(x)
        x = self.fc_final(x)
        return x     

def _test():
    from torchsummary import summary
    model = Inception_v3()
    model = model.cuda()
    summary(model,input_size=(3,299,299))

if __name__ == "__main__":
    _test()


# ------------------------------- mistakes ---------------------------------- #
# mis calculation after nn.faltten()
# no padding in inception stride = 2 module
# TODO: still more 3M than original paper
# TODO: check v3 init
# ---------------------------------- end ------------------------------------ #


# ------------------------------- background -------------------------------- #
#
# ---------------------------------- end ------------------------------------ #


# ---------------------------------- notes ---------------------------------- #
# parameters: 13M v1, 27M v3, BN+ReLUv3
# TODO: Training check: False
# ---------------------------------- end ------------------------------------ #


# ------------------------ Inception_v1 model summary ----------------------- #
# ----------------------------------------------------------------
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 64, 112, 112]           9,408
#               ReLU-2         [-1, 64, 112, 112]               0
#          MaxPool2d-3           [-1, 64, 56, 56]               0
#             Conv2d-4          [-1, 192, 56, 56]         110,592
#               ReLU-5          [-1, 192, 56, 56]               0
#                              ...
#           Conv2d-150            [-1, 128, 7, 7]         106,624
#             ReLU-151            [-1, 128, 7, 7]               0
# _InceptionV1Block-152           [-1, 1024, 7, 7]               0
# AdaptiveAvgPool2d-153           [-1, 1024, 1, 1]               0
#          Dropout-154           [-1, 1024, 1, 1]               0
#          Flatten-155                 [-1, 1024]               0
#           Linear-156                 [-1, 1000]       1,025,000
#          Softmax-157                 [-1, 1000]               0
# ================================================================
# Total params: 13,373,864
# Trainable params: 13,373,864
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 66.74
# Params size (MB): 51.02
# Estimated Total Size (MB): 118.33
# ---------------------------------- end ------------------------------------ #


# ------------------------ Inception_v3 model summary ----------------------- #
#         Layer (type)               Output Shape         Param #
# ================================================================
#             Conv2d-1         [-1, 32, 149, 149]             864
#        BatchNorm2d-2         [-1, 32, 149, 149]              64
#        BasicConv2d-3         [-1, 32, 149, 149]               0
#             Conv2d-4         [-1, 32, 147, 147]           9,216
#        BatchNorm2d-5         [-1, 32, 147, 147]              64
#        BasicConv2d-6         [-1, 32, 147, 147]               0
#                                 ...
# _InceptionV3Block_5-316           [-1, 2048, 8, 8]               0
# AdaptiveAvgPool2d-317           [-1, 2048, 1, 1]               0
#          Dropout-318           [-1, 2048, 1, 1]               0
#          Flatten-319                 [-1, 2048]               0
#           Linear-320                 [-1, 1000]       2,049,000
#          Softmax-321                 [-1, 1000]               0
# ================================================================
# Total params: 27,161,264
# Trainable params: 27,161,264
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 1.02
# Forward/backward pass size (MB): 245.12
# Params size (MB): 103.61
# Estimated Total Size (MB): 349.76
# ---------------------------------- end ------------------------------------ #