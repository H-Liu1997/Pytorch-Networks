# --------------------------------------------------------------------------- #
# ResNet, CVPR2016 bestpaper, https://arxiv.org/abs/1512.03385
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F
from config import cfg
from utils import load_cfg,model_complexity


__all__ = ['ResNet18','ResNet34','ResNet50','ResNet101','ResNet152']


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self,in_dim,out_dim,stride=1,op="A"):
        super(BasicBlock,self).__init__()
        self.subconv_1 = nn.Sequential(
            nn.Conv2d(in_dim,out_dim,3,stride,1,bias=False),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),)
        self.subconv_2 = nn.Sequential(
            nn.Conv2d(out_dim,out_dim,3,1,1,bias=False),
            nn.BatchNorm2d(out_dim))
        if in_dim == out_dim and stride == 1:
            self.downsample = nn.Sequential()
        elif op == 'A':
            self.downsample =LambdaLayer(lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, out_dim//4, out_dim//4), "constant", 0))
        else:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_dim,out_dim,1,stride,0,bias=False),
                nn.BatchNorm2d(out_dim),
            )
 
    def forward(self,input_):
        x_0 = self.subconv_1(input_)
        x_1 = self.subconv_2(x_0)
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
            self.maxpool_1 = nn.Sequential()
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
                nn.Linear(final_feature,cfg.CLASS_NUM),)
        else:
            self.avgpool_1 = nn.Sequential()
            self.fc_1 = nn.Sequential()
        self.logger = logger
        self.pretrained = cfg.PRETRAINED
        self._initialization()
    
    def _initialization(self):
        if self.pretrained is not False:
            self.modules.load_state_dict(model_zoo.load_url(model_urls[self.pretrained]))
            #TODO(liu):check it correct or not.
        else:
            for name, sub_module in self.named_modules():
                if isinstance(sub_module, nn.Conv2d) or isinstance(sub_module, nn.ConvTranspose2d) or \
                    isinstance(sub_module, nn.Linear):
                    nn.init.kaiming_normal_(sub_module.weight)
                    # nn.init.kaiming_normal_(sub_module.weight,mode='fan_out'
                    #                         ,nonlinearity='relu')
                    if self.logger is not None:
                        self.logger.info('init {}.weight as kaiming_normal_'.format(name))
                    if sub_module.bias is not None:
                        nn.init.constant_(sub_module.bias, 0.0)
                        if self.logger is not None:
                            self.logger.info('init {}.bias as 0'.format(name))
                # elif isinstance(sub_module, nn.BatchNorm2d):
                #     nn.init.constant_(sub_module.weight,1)
                #     nn.init.constant_(sub_module.bias,0)
                #     if self.logger is not None:
                #         self.logger.info('init {}.weight as constant_ 1'.format(name))
                #         self.logger.info('init {}.bias as constant_ 0'.format(name))
            
    def _make_layer(self,block,in_dim,out_dim,layer_num,stride):
        net_layers = []
        if layer_num == 0:
            return nn.Sequential()
        else:    
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


class ThreeDCNN(nn.Module):
    def __init__(self,cfg,logger):
        super(ThreeDCNN,self).__init__()
        self.res1 = ResNet(cfg,logger)
        self.res2 = ResNet(cfg,logger)
        self.res3 = ResNet(cfg,logger)
        self.getheatmap_1 = nn.Conv2d(128,19,1,1,0)
        self.getheatmap_2 = nn.Conv2d(128,19,1,1,0)
        self.getheatmap_3 = nn.Conv2d(128,19,1,1,0)

        self.getdepth_1 = nn.Conv2d(128,1,1,1,0)
        self.getdepth_2 = nn.Conv2d(128,1,1,1,0)
        self.getdepth_3 = nn.Conv2d(128,1,1,1,0)

        self.tdcnn1 = nn.Conv3d(19,128,3,1,1)#b,in,d,h,w,
        self.tdcnn2 = nn.Conv3d(128,128,3,1,1)
        self.maxpool3d_1 = nn.MaxPool3d(3,1,0)
        self.tdcnn3 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn331 = nn.Conv3d(128,128,3,1,1)


        self.tdcnn332 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn333 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn334 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn335 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn336 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn337= nn.Conv3d(128,128,3,1,1)
        self.tdcnn338 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn339 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn3310 = nn.Conv3d(128,128,3,1,1)
       


        self.tdcnn4 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn5 = nn.Conv3d(128,19,3,1,1)


        self.tdcnn6 = nn.Conv3d(1,128,3,1,1)#b,in,d,h,w,
        self.tdcnn7 = nn.Conv3d(128,128,3,1,1)
        self.maxpool3d_2 = nn.MaxPool3d(3,1,0)
        self.tdcnn8 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn88 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn9 = nn.Conv3d(128,128,3,1,1)
        self.tdcnn10 = nn.Conv3d(128,1,3,1,1)

    def forward(self,x):
        x1 = x[:,0,:,:,:]
        x2 = x[:,1,:,:,:]
        x3 = x[:,2,:,:,:]

        output1 = self.res1(x1)
        output2 = self.res2(x2)
        output3 = self.res3(x3)
        
        #print(output1.shape)
        de_output1 = self.getdepth_1(output1)
        de_output2 = self.getdepth_2(output2)
        de_output3 = self.getdepth_3(output3)

        he_output1 = self.getheatmap_1(output1)#(b,19,h,w)
        he_output2 = self.getheatmap_2(output2)
        he_output3 = self.getheatmap_3(output3)
     
        he_3d = torch.cat((he_output1.unsqueeze(2),
            he_output2.unsqueeze(2),
            he_output3.unsqueeze(2)),dim=2)#(b,19,3,h,w)
        de_3d = torch.cat((de_output1.unsqueeze(2),
            de_output2.unsqueeze(2),
            de_output3.unsqueeze(2)),dim=2)
        
        he_3d = self.tdcnn1(he_3d)
        he_3d = self.tdcnn2(he_3d)
        he_3d = self.maxpool3d_1(he_3d)
        he_3d = self.tdcnn3(he_3d)
        he_3d = self.tdcnn331(he_3d)

        he_3d = self.tdcnn332(he_3d)
        he_3d = self.tdcnn333(he_3d)
        he_3d = self.tdcnn334(he_3d)
        he_3d = self.tdcnn335(he_3d)
        he_3d = self.tdcnn336(he_3d)
        he_3d = self.tdcnn337(he_3d)
        he_3d = self.tdcnn338(he_3d)
        he_3d = self.tdcnn339(he_3d)
        he_3d = self.tdcnn3310(he_3d)

        he_3d = self.tdcnn4(he_3d)
        he_3d = self.tdcnn5(he_3d)

        de_3d = self.tdcnn6(de_3d)
        de_3d = self.tdcnn7(de_3d)
        de_3d = self.maxpool3d_2(de_3d)
        de_3d = self.tdcnn8(de_3d)
        de_3d = self.tdcnn88(de_3d)
        de_3d = self.tdcnn9(de_3d)
        de_3d = self.tdcnn10(de_3d)
        
        return de_3d, he_3d

        
if __name__ == "__main__":
    logger = load_cfg(cfg)
    model = ThreeDCNN(cfg.MODEL,logger).cuda()
    from ptflops import get_model_complexity_info
    flops, params = get_model_complexity_info(model,  (3,3,368,368), 
        as_strings=True, print_per_layer_stat=True)
    logger.info('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    logger.info('{:<30}  {:<8}'.format('Number of parameters: ', params))
    
    fakeinput = torch.ones((8,3,3,368,368)).cuda()
    output = model(fakeinput)
    mem = torch.cuda.memory_cached() / 1E9
    print(mem)

    










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