# ------------------------------------------------------------------------------
# This is multi-stage FPN for pose estimation
# single stage paper:
# https://arxiv.org/abs/1807.04067 ECCV2018, no one can reproduce its mAP
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torch.nn import init


def network_cli(parser):
    ''' network config
        1. paf and heatmap nums
        2. weight path
    '''
    print("using fpn_net success")
    group = parser.add_argument_group('network')
    group.add_argument('--heatmap_num', default=19, type=int)
    group.add_argument('--paf_num', default=38, type=int)
    group.add_argument('--paf_stage', default=4, type=int)
    group.add_argument('--weight_res50', default='https://download.pytorch.org/models/resnet50-19c8e357.pth')
    group.add_argument('--weight_res101', default='https://download.pytorch.org/models/resnet101-5d3b4d8f.pth')
    

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class FPN(nn.Module):
    def __init__(self, block, num_blocks):
        super(FPN, self).__init__()
        self.in_planes = 64

        self.toc1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        #self.pooling1 = nn.max_pool2d(2,2,0)

        # Bottom-up layers
        self.c1toc2 = self._make_layer(block,  64, num_blocks[0], stride=1) #c2
        self.c2toc3 = self._make_layer(block, 128, num_blocks[1], stride=2) #c3
        self.c3toc4 = self._make_layer(block, 256, num_blocks[2], stride=2) #c4
        self.c4toc5 = self._make_layer(block, 512, num_blocks[3], stride=2) #c5

        self.c5tom5   = nn.Conv2d(2048,256, kernel_size=1, stride=1, padding=0)
        self.c4toc4_ = nn.Conv2d(1024, 256, kernel_size=1, stride=1, padding=0)  # c5 -> c5'
        self.c3toc3_ = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)  # c4 -> c4'
        self.c2toc2_= nn.Conv2d(256, 256, kernel_size=1, stride=1, padding=0)  # c3 -> c3'
       

        # one part 
        self.m5top5   = nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1)
        self.m4top4   = nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1)
        self.m3top3   = nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1)
        self.m2top2   = nn.Conv2d(256,128, kernel_size=3, stride=1, padding=1)

        self.m5top52   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.m4top42   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.m3top32   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.m2top22   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        # one part finish
        
        # one part
        self.m5tok5   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)
        self.m4tok4   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)
        self.m3tok3   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)
        self.m2tok2   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)

        self.m5tok52   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.m4tok42   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.m3tok32   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.m2tok22   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)

        # one part finish

        # one part
        self.p50top51   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)
        self.p40top41   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)
        self.p30top31   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)
        self.p20top21   = nn.Conv2d(256 + 38,128, kernel_size=3, stride=1, padding=1)

        self.p50top512   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.p40top412   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.p30top312   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        self.p20top212   = nn.Conv2d(128,128, kernel_size=3, stride=1, padding=1)
        # one part finish

        self.psumsmoothp   = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.psumsmoothp1   = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.psumsmoothk   = nn.Conv2d(512,512, kernel_size=3, stride=1, padding=1)
        self.psumtopaf   = nn.Conv2d(512,38, kernel_size=1, stride=1, padding=0)
        self.psumtopaf1   = nn.Conv2d(512,38, kernel_size=1, stride=1, padding=0)
        self.psumtoheat   = nn.Conv2d(512,19, kernel_size=1, stride=1, padding=0)
        

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def initi(self):
        init.normal_(self.self.c5tom5.weight, std=0.01)
        init.normal_(self.self.c4toc4_.weight, std=0.01)
        init.normal_(self.self.c3toc3_.weight, std=0.01)
        init.normal_(self.self.c2toc2_.weight, std=0.01)

        init.normal_(self.self.m5top5.weight, std=0.01)
        init.normal_(self.self.m4top4.weight, std=0.01)
        init.normal_(self.self.m3top3.weight, std=0.01)
        init.normal_(self.self.m2top2.weight, std=0.01)

        init.normal_(self.self.m5top52.weight, std=0.01)
        init.normal_(self.self.m4top42.weight, std=0.01)
        init.normal_(self.self.m3top32.weight, std=0.01)
        init.normal_(self.self.m2top22.weight, std=0.01)

        init.normal_(self.self.m5tok5.weight, std=0.01)
        init.normal_(self.self.m4tok4.weight, std=0.01)
        init.normal_(self.self.m3tok3.weight, std=0.01)
        init.normal_(self.self.m2tok2.weight, std=0.01)

        init.normal_(self.self.m5tok52.weight, std=0.01)
        init.normal_(self.self.m4tok42.weight, std=0.01)
        init.normal_(self.self.m3tok32.weight, std=0.01)
        init.normal_(self.self.m2tok22.weight, std=0.01)

        init.normal_(self.self.p50top51.weight, std=0.01)
        init.normal_(self.self.p40top41.weight, std=0.01)
        init.normal_(self.self.p30top31.weight, std=0.01)
        init.normal_(self.self.p20top21.weight, std=0.01)

        init.normal_(self.self.p50top512.weight, std=0.01)
        init.normal_(self.self.p40top412.weight, std=0.01)
        init.normal_(self.self.p30top312.weight, std=0.01)
        init.normal_(self.self.p20top212.weight, std=0.01)

        init.normal_(self.self.psumsmoothp.weight, std=0.01)
        init.normal_(self.self.psumsmoothp1.weight, std=0.01)
        init.normal_(self.self.psumsmoothk.weight, std=0.01)
        init.normal_(self.self.psumtopaf.weight, std=0.01)
        init.normal_(self.self.psumtopaf1.weight, std=0.01)
        init.normal_(self.self.psumtoheat.weight, std=0.01)




    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: top feature map to be upsampled.
          y: lateral feature map.
        Returns:
          added feature map.
        '''
        _,_,H,W = y.size()
        return F.upsample(x, size=(H,W), mode='nearest', align_corners=None) + y  # bilinear, False

    def forward(self, x):
        # Bottom-up

        save_for_loss = []

        c1 = F.relu(self.bn1(self.toc1(x)))
        c1 = F.max_pool2d(c1, kernel_size=2, stride=2, padding=0)
        c2 = F.relu(self.c1toc2(c1))
        c3 = F.relu(self.c2toc3(c2))
        c4 = F.relu(self.c3toc4(c3))
        c5 = F.relu(self.c4toc5(c4))

        m5 = F.relu(self.c5tom5(c5))
        c4_ = F.relu(self.c4toc4_(c4))
        c3_ = F.relu(self.c3toc3_(c3))
        c2_ = F.relu(self.c2toc2_(c2))

        m4 = self._upsample_add(m5,c4_)
        m3 = self._upsample_add(m4,c3_)
        m2 = self._upsample_add(m3,c2_)

        p2 = F.relu(self.m2top2(m2))
        p3 = F.relu(self.m3top3(m3))
        p4 = F.relu(self.m4top4(m4))
        p5 = F.relu(self.m5top5(m5))

        p2_ = F.relu(self.m2top22(p2))
        p3_ = F.relu(self.m3top32(p3))
        p4_ = F.relu(self.m4top42(p4))
        p5_ = F.relu(self.m5top52(p5))

        p2_s = F.upsample(p2_,size=(120,120), mode='nearest', align_corners=None)
        p3_s = F.upsample(p3_,size=(120,120), mode='nearest', align_corners=None)
        p4_s = F.upsample(p4_,size=(120,120), mode='nearest', align_corners=None)
        p5_s = F.upsample(p5_,size=(120,120), mode='nearest', align_corners=None)

        psum = torch.cat([p2_s,p3_s,p4_s,p5_s],1)
        psum_ = F.relu(self.psumsmoothp(psum))
        paf_0_0 = self.psumtopaf(psum_)
        save_for_loss.append(paf_0_0)

        paf_1_0 = F.interpolate(paf_0_0, size=(60,60),mode='bilinear', align_corners=None)
        paf_2_0 = F.interpolate(paf_0_0, size=(30,30),mode='bilinear', align_corners=None)
        paf_3_0 = F.interpolate(paf_0_0, size=(15,15),mode='bilinear', align_corners=None)

        p_in_2 = torch.cat([paf_0_0,m2],1)
        p_in_3 = torch.cat([paf_1_0,m3],1)
        p_in_4 = torch.cat([paf_2_0,m4],1)
        p_in_5 = torch.cat([paf_3_0,m5],1)

        p21 = F.relu(self.p20top21(p_in_2))
        p31 = F.relu(self.p30top31(p_in_3))
        p41 = F.relu(self.p40top41(p_in_4))
        p51 = F.relu(self.p50top51(p_in_5))

        p21_ = F.relu(self.p20top212(p21))
        p31_ = F.relu(self.p30top312(p31))
        p41_ = F.relu(self.p40top412(p41))
        p51_ = F.relu(self.p50top512(p51))

        p21_s = F.upsample(p21_,size=(120,120), mode='nearest', align_corners=None)
        p31_s = F.upsample(p31_,size=(120,120), mode='nearest', align_corners=None)
        p41_s = F.upsample(p41_,size=(120,120), mode='nearest', align_corners=None)
        p51_s = F.upsample(p51_,size=(120,120), mode='nearest', align_corners=None)

        psum1 = torch.cat([p21_s,p31_s,p41_s,p51_s],1)
        psum1_ = F.relu(self.psumsmoothp1(psum1))
        paf_0_1 = self.psumtopaf1(psum1_)

        paf_1_1 = F.interpolate(paf_0_1, size=(60,60),mode='bilinear', align_corners=None)
        paf_2_1 = F.interpolate(paf_0_1, size=(30,30),mode='bilinear', align_corners=None)
        paf_3_1 = F.interpolate(paf_0_1, size=(15,15),mode='bilinear', align_corners=None)
        save_for_loss.append(paf_0_1)

        k_in_2 = torch.cat([paf_0_1,m2],1)
        k_in_3 = torch.cat([paf_1_1,m3],1)
        k_in_4 = torch.cat([paf_2_1,m4],1)
        k_in_5 = torch.cat([paf_3_1,m5],1)

        k2 = F.relu(self.m2tok2(k_in_2))
        k3 = F.relu(self.m3tok3(k_in_3))
        k4 = F.relu(self.m4tok4(k_in_4))
        k5 = F.relu(self.m5tok5(k_in_5))

        k2_ = F.relu(self.m2tok22(k2))
        k3_ = F.relu(self.m3tok32(k3))
        k4_ = F.relu(self.m4tok42(k4))
        k5_ = F.relu(self.m5tok52(k5))

        k2_s = F.upsample(k2_,size=(120,120), mode='nearest', align_corners=None)
        k3_s = F.upsample(k3_,size=(120,120), mode='nearest', align_corners=None)
        k4_s = F.upsample(k4_,size=(120,120), mode='nearest', align_corners=None)
        k5_s = F.upsample(k5_,size=(120,120), mode='nearest', align_corners=None)

        ksum = torch.cat([k2_s,k3_s,k4_s,k5_s],1)
        ksum_ = F.relu(self.psumsmoothk(ksum))
        heat = self.psumtoheat(ksum_)

        save_for_loss.append(heat)

        return (paf_1_0,heat), save_for_loss

def FPN50():
    # [3,4,6,3] -> resnet50
    return FPN(Bottleneck, [3,4,6,3])

def FPN101():
    # [3,4,23,3] -> resnet101
    return FPN(Bottleneck, [3,4,23,3])


if __name__ == "__main__":
    import argparse
    from torchsummary import summary

    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    network_cli(parser)
    args = parser.parse_args()
    model = FPN50()
    model = model.cuda()
    summary(model,input_size=(3,480,480))
