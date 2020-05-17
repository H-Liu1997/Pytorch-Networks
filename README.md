<div align=center><img width="300" height="150" src="docs/images/cat.jpg"/>
  
# Pytorch Networks

**The Pytorch implementations of famous basic networks. Unscheduled update.**

[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HaiyangLiu1997/Pytorch-Networks/blob/master/LICENSE)
[![stars](https://img.shields.io/github/stars/HaiyangLiu1997/Pytorch-Networks.svg)](https://github.com/HaiyangLiu1997/Pytorch-Networks/stargazers)
</div>

## Accuracy Report
|  Model   |   Dataset   |  #Params  |Epochs| Test Prec(Paper)  | Test Prec(This impl) |
|----------|:-----------:|:---------:|:-----:|:-:|:------:|
|[ResNet-20](https://github.com/HaiyangLiu1997/Pytorch-Networks/tree/master/exps/res_18_v3_la)|  CIFAR-10  | 0.27M | 140,180,200 |91.25%  |  **92.02%**   |
|[ResNet-32](https://github.com/HaiyangLiu1997/Pytorch-Networks/tree/master/exps/res_34_la)|  CIFAR-10  | 0.46M |140,180,200| 92.49%  |  **92.61%**   |
|[ResNet-44](https://github.com/HaiyangLiu1997/Pytorch-Networks/tree/master/exps/res_44_100_150_swaug)|  CIFAR-10  | 0.66M | 100,150,200|**92.83%**  |  92.46%   |
|[ResNet-56](https://github.com/HaiyangLiu1997/Pytorch-Networks/tree/master/exps/res_56_100_150_swaug)|  CIFAR-10  | 0.85M | 100,150,200|93.03%  |  **93.22%**   |
|[ResNet-110](https://github.com/HaiyangLiu1997/Pytorch-Networks/tree/master/exps/res_110_la)|  CIFAR-10  | 1.73M | 82,123,164 |**93.57%**  |  93.40%   |
|[ResNet-18(A)](https://github.com/HaiyangLiu1997/Pytorch-Networks/tree/master/exps/res_18_100_150)|  CIFAR-10  | 11.0M |100,150,200|- | **93.54%**   |
|[ResNet-18(B)](https://github.com/HaiyangLiu1997/Pytorch-Networks/tree/master/exps/res_18_100_150_b)|  CIFAR-10  | 11.17M |80,110,120|- | **94.51%**  |
|ResNet-50|  CUB-200  | 23.92M |26,36,40|- | **81.74%**  |
|ResNeXt-50| CUB-200 | 23.39M |26,36,40|-|**82.70%**  |
|RegNetX-4.0|  CUB-200  | 21.03M |26,36,40|- | **84.31%**  |
|RegNetY-8.0|  CUB-200  | 37.57M |26,36,40|- | **84.40%**  |
|RegNetY-8.0|  CUB-200  | 37.57M |cos60+wp|- | **84.86%**  |
|RegNetY-32.0|  CUB-200  | 142.08M |cos60|- | **85.23%**  |
|EfficientNet-B2|  CUB-200  | 7.98M |26,36,40|- | **82.60%**  |
|RegY32+EB2+NeXt50|  CUB-200  | - |-|- | **87.31%**  |




**For backbone networks**

Most of backbone networks already have pytorch official version(ResNet.etc), my implementations have a little diffience with them because of my programming habits

**For other networks**

Some networks don't have offical pytorch version for several reasons(author didn't public the code.etc), my implementations are totally original reproductions

## Network list
### Backbone(chronological order)
Common Type:
- [x] AlexNet                 2012
- [x] NIN                     2013
- [x] VGG                     2014
- [x] Inception/GoogLeNet     2014
- [x] InceptionV2/V3          2015
- [x] InceptionV4/Res         2016
- [ ] Xception                2016
- [x] ResNet                  2015
- [x] ResNetV2                2016
- [x] ResNeXt                 2016
- [ ] Res2Net                 2019
- [x] DenseNet                2016
- [x] SENet                   2017
- [ ] NASNet                  2017
- [ ] AmoebaNet               2018
- [x] MnasNet                 2018
- [x] EfficientNet            2019

Light Type:
- [x] ShuffleNet              2017
- [x] MobileNet               2017

Other:
- [x] BasicGNN

### Pose Estimation(chronological order)
- [x] OpenPose                2015
- [x] Hourglass               2015
- [x] GNNlikeCNN              2015
- [x] SimpleBaseline          2017
- [x] CPN                     2017
- [x] OpenPose                2017
- [x] FPNPoseNet              2017
- [ ] HRNet                   2018
- [x] CPN+GNN                 2018
- [ ] Multi-CPN               2019

### Detection
- [x] Darknet-19              2016
- [x] Darknet-53              2018 

## Future Work: Pytorch loss
Will release a new repertory in the future(not in current repertory)
- [ ] Centerloss              2015
- [x] Focusloss               2016 
- [x] labelsmooth
...

