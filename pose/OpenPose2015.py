# ------------------------------------------------------------------------------
# The old OpenPose Pytorch Implementation 
# https://arxiv.org/abs/1611.08050
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------
import torch.nn as nn
import torch
from torch.nn import init


def network_cli(parser):
    ''' network config
        1. paf and heatmap nums
        2. weight path
    '''
    print("using cmu_old_net success")
    group = parser.add_argument_group('network')
    group.add_argument('--heatmap_num', default=19, type=int)
    group.add_argument('--paf_num', default=38, type=int)
    #group.add_argument('--paf_stage', default=4, type=int)
    group.add_argument('--weight_vgg19', default='https://download.pytorch.org/models/vgg19-dcbb9e9d.pth')

    
class Debugnetwork(nn.Module):
    '''
    '''
    def __init__(self,args):
        super(Debugnetwork,self).__init__()
        self.block_0 = VGG_19(3)

    def forward(self,input_):
        output = self.block_0(input_)
        return output


class CMUnetwork(nn.Module):
    '''
    '''
    def __init__(self,args):
        super(CMUnetwork,self).__init__()
        self.block_0 = VGG_19(3)
        self.ch_sum = 128+args.paf_num+args.heatmap_num
        self.block_1_1 = stage_1_block(128,args.paf_num)
        self.block_1_2 = stage_1_block(128,args.heatmap_num) 

        self.block_2_1 = stage_n_block(self.ch_sum,args.paf_num)
        self.block_2_2 = stage_n_block(self.ch_sum,args.heatmap_num)

        self.block_3_1 = stage_n_block(self.ch_sum,args.paf_num)
        self.block_3_2 = stage_n_block(self.ch_sum,args.heatmap_num)

        self.block_4_1 = stage_n_block(self.ch_sum,args.paf_num)
        self.block_4_2 = stage_n_block(self.ch_sum,args.heatmap_num)

        self.block_5_1 = stage_n_block(self.ch_sum,args.paf_num)
        self.block_5_2 = stage_n_block(self.ch_sum,args.heatmap_num)

        self.block_6_1 = stage_n_block(self.ch_sum,args.paf_num)
        self.block_6_2 = stage_n_block(self.ch_sum,args.heatmap_num)

    def forward(self,input_):
        save_for_loss =[]
        output_0 = self.block_0(input_)
        output_1_1 = self.block_1_1(output_0)
        output_1_2 = self.block_1_2(output_0)
        save_for_loss.append(output_1_1)
        save_for_loss.append(output_1_2)
        #print('1-2:',output_1_1.size())
        #print('1-1:',output_1_2.size())
        output_1_sum = torch.cat([output_1_1,output_1_2,output_0],1)
        #print('1-sum:',output_1_sum.size())
        output_2_1 = self.block_2_1(output_1_sum)
        output_2_2 = self.block_2_2(output_1_sum)
        save_for_loss.append(output_2_1)
        save_for_loss.append(output_2_2)
        output_2_sum = torch.cat([output_2_1,output_2_2,output_0],1)

        output_3_1 = self.block_3_1(output_2_sum)
        output_3_2 = self.block_3_2(output_2_sum)
        save_for_loss.append(output_3_1)
        save_for_loss.append(output_3_2)
        output_3_sum = torch.cat([output_3_1,output_3_2,output_0],1)

        output_4_1 = self.block_4_1(output_3_sum)
        output_4_2 = self.block_4_2(output_3_sum)
        save_for_loss.append(output_4_1)
        save_for_loss.append(output_4_2)
        output_4_sum = torch.cat([output_4_1,output_4_2,output_0],1)

        output_5_1 = self.block_5_1(output_4_sum)
        output_5_2 = self.block_5_2(output_4_sum)
        save_for_loss.append(output_5_1)
        save_for_loss.append(output_5_2)
        output_5_sum = torch.cat([output_5_1,output_5_2,output_0],1)

        output_6_1 = self.block_6_1(output_5_sum)
        output_6_2 = self.block_6_2(output_5_sum)
        save_for_loss.append(output_6_1)
        save_for_loss.append(output_6_2)

        return (output_6_1,output_6_2),save_for_loss                                                                                                                                         

class conv(nn.Module):
    '''
    n*n conv with relu
    '''
    def __init__(self,in_dim,out_dim,kernal_size,stride,padding):
        super(conv,self).__init__()
        self.con_layer = nn.Conv2d(in_dim,out_dim,kernal_size,stride,padding)
        self.relu = nn.ReLU(inplace=True)
        self.initi()
    
    def forward(self,input_):
        output = self.con_layer(input_)
        output = self.relu(output)
        return output
    def initi(self):
        #init.kaiming_normal_(self.con_layer.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.normal_(self.con_layer.weight, std=0.01)
        if self.con_layer.bias is not None:  
            init.constant_(self.con_layer.bias, 0.0)



class stage_1_block(nn.Module):
    '''
    stage 1 only 5 layers and the kernal size is 5
    last layer don't have relu
    '''
    def __init__(self,input_dim,output_dim):
        super(stage_1_block,self).__init__()
        self.conv1 = conv(input_dim,128,3,1,1)
        self.conv2 = conv(128,128,3,1,1)
        self.conv3 = conv(128,128,3,1,1)
        self.conv4 = conv(128,512,1,1,0)
        self.conv5 = nn.Conv2d(512,output_dim,1,1,0)
        
        self.initi()
    
    def forward(self, input_):
        output = self.conv1(input_)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        return output

    def initi(self):
        #init.kaiming_normal_(self.conv5.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.normal_(self.conv5.weight, std=0.01)
        if self.conv5.bias is not None:  
            init.constant_(self.conv5.bias, 0.0)


class stage_n_block(nn.Module):
    '''
    stage n only 7 layers and the kernal size is 7
    last layer don't have relu
    '''
    def __init__(self,input_dim,output_dim):
        super(stage_n_block,self).__init__()
        self.conv1 = conv(input_dim,128,7,1,3)
        self.conv2 = conv(128,128,7,1,3)
        self.conv3 = conv(128,128,7,1,3)
        self.conv4 = conv(128,128,7,1,3)
        self.conv5 = conv(128,128,7,1,3)
        self.conv6 = conv(128,128,1,1,0)
        self.conv7 = nn.Conv2d(128,output_dim,1,1,0)
        
        self.initi()
    
    def forward(self, input_):
        output = self.conv1(input_)
        output = self.conv2(output)
        output = self.conv3(output)
        output = self.conv4(output)
        output = self.conv5(output)
        output = self.conv6(output)
        output = self.conv7(output)
        return output

    def initi(self):
        #init.kaiming_normal_(self.conv7.weight, a=0, mode='fan_in', nonlinearity='relu')
        init.normal_(self.conv7.weight, std=0.01)
        if self.conv7.bias is not None:  
            init.constant_(self.conv7.bias, 0.0)



class VGG_19(nn.Module):
    '''
    VGG_19 first 10 layers
    11 and 12 by CMU
    '''
    def __init__(self,input_dim):
        super(VGG_19,self).__init__()
        self.conv1_1 = conv(input_dim,64,3,1,1)
        self.conv1_2 = conv(64,64,3,1,1)
        self.pooling_1 = nn.MaxPool2d(2,2,0)
        self.conv2_1 = conv(64,128,3,1,1)
        self.conv2_2 = conv(128,128,3,1,1)
        self.pooling_2 = nn.MaxPool2d(2,2,0)
        self.conv3_1 = conv(128,256,3,1,1)
        self.conv3_2 = conv(256,256,3,1,1)
        self.conv3_3 = conv(256,256,3,1,1)
        self.conv3_4 = conv(256,256,3,1,1)
        self.pooling_3 = nn.MaxPool2d(2,2,0)
        self.conv4_1 = conv(256,512,3,1,1)
        self.conv4_2 = conv(512,512,3,1,1)
        self.conv4_3 = conv(512,256,3,1,1)
        self.conv4_4 = conv(256,128,3,1,1)

    def forward(self,input_):
        output = self.conv1_1(input_)
        output = self.conv1_2(output)
        output = self.pooling_1(output)
        output = self.conv2_1(output)
        output = self.conv2_2(output)
        output = self.pooling_2(output)
        output = self.conv3_1(output)
        output = self.conv3_2(output)
        output = self.conv3_3(output)
        output = self.conv3_4(output)
        output = self.pooling_3(output)
        output = self.conv4_1(output)
        output = self.conv4_2(output)
        output = self.conv4_3(output)
        output = self.conv4_4(output)
        return output





if __name__ == "__main__":
    import argparse
    from torchsummary import summary

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    network_cli(parser)
    args = parser.parse_args()
    model = Debugnetwork(args)
    summary(model,input_size=(3,368,368))
