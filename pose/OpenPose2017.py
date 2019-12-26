# ------------------------------------------------------------------------------
# The newest OpenPose Pytorch Implementation
# https://arxiv.org/abs/1812.08008
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
from torch.nn import init

def network_cli(parser):
    ''' network config
        1. paf and heatmap nums
        2. weight path
    '''
    print("using cmu_new_net success")
    group = parser.add_argument_group('network')
    group.add_argument('--heatmap_num', default=19, type=int)
    group.add_argument('--paf_num', default=38, type=int)
    group.add_argument('--paf_stage', default=4, type=int)
    group.add_argument('--weight_vgg19', default='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')
    
class CMUnetwork(nn.Module):
    ''' the newest cmu network'''

    def __init__ (self,args):
        # already finish the init_weight in each block
        super(CMUnetwork, self).__init__()
        self.state_0 = VGG_block()
        self.state_1 = state_1_block(128, args.paf_num)
        self.state_2 = state_n_block(128+args.paf_num,args.paf_num)
        self.state_3 = state_n_block(128+args.paf_num,args.paf_num)
        self.state_4 = state_n_block(128+args.paf_num,args.paf_num)
        self.state_5 = state_1_block(128+args.paf_num,args.heatmap_num)
        self.state_6 = state_n_block(128+args.heatmap_num+args.paf_num,args.heatmap_num)

    def forward(self,input_0):

        saved_for_loss = []

        output_0 = self.state_0(input_0)
        output_1 = self.state_1(output_0)
        input_2  = torch.cat([output_0,output_1],1)
        saved_for_loss.append(output_1)

        output_2 = self.state_2(input_2)
        input_3  = torch.cat([output_0,output_2],1)
        saved_for_loss.append(output_2)

        output_3 = self.state_3(input_3)
        input_4  = torch.cat([output_0,output_3],1)
        saved_for_loss.append(output_3)

        output_4 = self.state_4(input_4)
        input_5  = torch.cat([output_0,output_4],1)
        saved_for_loss.append(output_4)

        output_5 = self.state_5(input_5)
        input_6  = torch.cat([output_0,output_5,output_4],1)
        saved_for_loss.append(output_5)

        output_6 = self.state_6(input_6)
        saved_for_loss.append(output_6)

        return (output_4,output_6), saved_for_loss


class dense_block(nn.Module):
    '''1. basic dense block of the new openpose
       2. add conv1,2,3 output together to output
       3. default kernal_size = 3,bias = true
    '''

    def __init__(self, in_dim, out_dim):
        super(dense_block, self).__init__()
        # default inplace = False for ReLU
        
        self.conv1 = nn.Sequential( nn.Conv2d(in_dim, 128, 3, 1, 1),
                                    nn.PReLU(num_parameters=128))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),       
                                    nn.PReLU(num_parameters=128))
        self.conv3 = nn.Sequential(nn.Conv2d(128, 128, 3, 1, 1),  
                                  nn.PReLU(num_parameters=128))
                                   
        self.initialize_weight()
        

    def forward(self,input_1):
        # debug = True
        output_1 = self.conv1(input_1)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        output = torch.cat([output_1,output_2,output_3],1)
        
        return output
    
    def initialize_weight(self):
        for m in self.modules():
            #print('need check init')
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                #init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            else: 
                try:init.constant_(m.weight,0.0)
                except:pass

class dense_block_0(nn.Module):
    '''1. basic dense block of the new openpose
       2. add conv1,2,3 output together to output
       3. default kernal_size = 3,bias = true
    '''

    def __init__(self, in_dim, out_dim):
        super(dense_block_0, self).__init__()
        # default inplace = False for ReLU
        
        self.conv1 = nn.Sequential( nn.Conv2d(in_dim, 96, 3, 1, 1),
                                    nn.PReLU(num_parameters = 96))
        self.conv2 = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1),
                                   nn.PReLU(num_parameters = 96))
        self.conv3 = nn.Sequential(nn.Conv2d(96, 96, 3, 1, 1),
                                   nn.PReLU(num_parameters = 96))
        self.initialize_weight()
        

    def forward(self,input_1):
        # debug = True
        output_1 = self.conv1(input_1)
        output_2 = self.conv2(output_1)
        output_3 = self.conv3(output_2)
        output = torch.cat([output_1,output_2,output_3],1)

        return output
    
    def initialize_weight(self):
        for m in self.modules():
            #print('need check init')
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                #init.normal_(m.weight, std = 0.01)
                if m.bias is not None:
                    init.constant_(m.bias, 0.0)
            else: 
                try:init.constant_(m.weight,0.0)
                except:pass
    
    
        
class state_n_block(nn.Module):
    ''' 1. for state 1 in_dim = 128
        2. for other state depend on the paf and heatmap channels
    '''

    def __init__(self, in_dim, out_dim):
        # 384 = 128 *3
        super(state_n_block, self).__init__()
        self.block1 = dense_block(in_dim,384)
        self.block2 = dense_block(384,128)
        self.block3 = dense_block(384,128)
        self.block4 = dense_block(384,128)
        self.block5 = dense_block(384,128)
        self.conv1  = nn.Sequential(nn.Conv2d(384, 512, 1, 1, 0),
                                    nn.PReLU(num_parameters = 512))
        self.conv2  = nn.Conv2d(512,out_dim,1,1,0)
        self.initialize_weight()

    def forward(self,input_1):
        ''' inplace the midresult '''
        output_1 = self.block1(input_1)
        output_1 = self.block2(output_1)
        output_1 = self.block3(output_1)                     
        output_1 = self.block4(output_1)
        output_1 = self.block5(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1
    
    def initialize_weight(self):
        '''init 1*1 conv block
        '''
        init.xavier_normal_(self.conv1[0].weight)
        #init.xavier_normal_(self.conv1[1].weight)
        init.xavier_normal_(self.conv2.weight)
        init.constant_(self.conv1[1].weight, 0.0)
        #init.normal_(self.conv1[0].weight, std =0.01)
        init.constant_(self.conv1[0].bias, 0.0)
        #init.normal_(self.conv2.weight, std =0.01)
        init.constant_(self.conv2.bias, 0.0)

class state_1_block(nn.Module):
    ''' 1. for state 1 in_dim = 128
        2. for other state depend on the paf and heatmap channels
    '''

    def __init__(self, in_dim, out_dim):
        # 384 = 128 *3
        super(state_1_block, self).__init__()
        self.block1 = dense_block_0(in_dim,288)
        self.block2 = dense_block_0(288,96)
        self.block3 = dense_block_0(288,96)
        self.block4 = dense_block_0(288,96)
        self.block5 = dense_block_0(288,96)
        self.conv1  = nn.Sequential(nn.Conv2d(288, 256, 1, 1, 0),
                                        nn.PReLU(num_parameters = 256))
        self.conv2  = nn.Conv2d(256,out_dim,1,1,0)
        self.initialize_weight()

    def forward(self,input_1):
        ''' inplace the midresult '''
        output_1 = self.block1(input_1)
        output_1 = self.block2(output_1)
        output_1 = self.block3(output_1)                     
        output_1 = self.block4(output_1)
        output_1 = self.block5(output_1)
        output_1 = self.conv1(output_1)
        output_1 = self.conv2(output_1)
        return output_1

    def initialize_weight(self):
        '''init 1*1 conv block
        '''
        init.xavier_normal_(self.conv1[0].weight)
        #init.xavier_normal_(self.conv1[1].weight)
        init.xavier_normal_(self.conv2.weight)
        
        #init.normal_(self.conv1[0].weight, std =0.01)
        init.constant_(self.conv1[1].weight, 0.0)
        init.constant_(self.conv1[0].bias, 0.0)
        #init.normal_(self.conv2.weight, std =0.01)
        init.constant_(self.conv2.bias, 0.0)


class VGG_block(nn.Module):
    ''' 1. default have the bias
        2. using ReLU and 3 * max pooling
        3. 10 layers of VGG original
        4. 2 extra layers by CMU
        5. default in_dim = 3,out_dim = 128
        6. all kernal_size = 3, stride = 1
    '''

    def __init__(self, in_dim = 3, out_dim = 128):
        super(VGG_block, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, 3, 1, 1)
        self.relu1_1 = nn.ReLU(inplace = True)            
        self.conv1_2 = nn.Conv2d(64, 64, 3, 1, 1)
        self.relu1_2 = nn.ReLU(inplace = True)
        self.pool_1 = nn.MaxPool2d(2, 2, 0)
        self.conv2_1 = nn.Conv2d(64, 128, 3, 1, 1)
        self.relu2_1 = nn.ReLU(inplace = True)
        self.conv2_2 = nn.Conv2d(128, 128, 3, 1, 1)
        self.relu2_2 = nn.ReLU(inplace = True)
        self.pool_2 = nn.MaxPool2d(2, 2, 0)
        self.conv3_1 = nn.Conv2d(128, 256, 3, 1, 1)
        self.relu3_1 = nn.ReLU(inplace = True)
        self.conv3_2 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_2 = nn.ReLU(inplace = True)
        self.conv3_3 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_3 = nn.ReLU(inplace = True)
        self.conv3_4 = nn.Conv2d(256, 256, 3, 1, 1)
        self.relu3_4 = nn.ReLU(inplace = True)
        self.pool_3 = nn.MaxPool2d(2, 2, 0)
        self.conv4_1 = nn.Conv2d(256, 512, 3, 1, 1)
        self.relu4_1 = nn.ReLU(inplace = True)
        self.conv4_2 = nn.Conv2d(512, 512, 3, 1, 1)
        self.relu4_2 = nn.PReLU(num_parameters=512)
        self.conv4_3_cmu = nn.Conv2d(512, 256, 3, 1, 1)
        #TODO: check the init of prelu in openpose
        self.relu4_3 = nn.PReLU(num_parameters=256)
        self.conv4_4_cmu = nn.Conv2d(256, 128, 3, 1, 1)
        self.relu4_4 = nn.PReLU(num_parameters=128)
        self.initilization()
                                
    def forward(self,input_1):
        '''inplace middle result '''
        #print("before_vgg",input_1.size())
        output_1 = self.conv1_1(input_1)
        output_1 = self.relu1_1(output_1)
        output_1 = self.conv1_2(output_1)
        output_1 = self.relu1_2(output_1)
        output_1 = self.pool_1(output_1)                     
        output_1 = self.conv2_1(output_1)
        output_1 = self.relu2_1(output_1)
        output_1 = self.conv2_2(output_1)
        output_1 = self.relu2_2(output_1)
        output_1 = self.pool_2(output_1)
        output_1 = self.conv3_1(output_1)
        output_1 = self.relu3_1(output_1)
        output_1 = self.conv3_2(output_1)
        output_1 = self.relu3_2(output_1)
        output_1 = self.conv3_3(output_1)
        output_1 = self.relu3_3(output_1)
        output_1 = self.conv3_4(output_1)
        output_1 = self.relu3_4(output_1)
        output_1 = self.pool_3(output_1)
        output_1 = self.conv4_1(output_1)
        output_1 = self.relu4_1(output_1)
        output_1 = self.conv4_2(output_1)
        output_1 = self.relu4_2(output_1)
        output_1 = self.conv4_3_cmu(output_1)
       
        output_1 = self.relu4_3(output_1)
        output_1 = self.conv4_4_cmu(output_1)
       
        output_1 = self.relu4_4(output_1)
        #print("after_vgg",input_1.size())
        return output_1
    
    def initilization(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.xavier_normal_(m.weight)
                #init.normal_(m.weight, std=0.01)
                if m.bias is not None:  
                    init.constant_(m.bias, 0.0)
            else:
                try:init.constant_(m.weight,0.0)
                except:pass

 

    






    
