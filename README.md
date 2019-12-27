<div align=center><img width="300" height="150" src="docs/images/cat.jpg"/>
  
# Pytorch Networks

**The Pytorch implementations of famous basic networks. Unscheduled update.**

[![license](https://img.shields.io/badge/License-MIT-blue.svg)](https://github.com/HaiyangLiu1997/Pytorch-Networks/blob/master/LICENSE)
[![stars](https://img.shields.io/github/stars/HaiyangLiu1997/Pytorch-Networks.svg)](https://github.com/HaiyangLiu1997/Pytorch-Networks/stargazers)
</div>

## How to use it
All networks already pass forward test

**For backbone networks**

Most of backbone networks already have pytorch official version(ResNet.etc), my implementations have a little diffience with them because of my programming habits

**For other networks**

Some networks don't have offical pytorch version for several reasons(author didn't public the code.etc), my implementations are totally original reproductions

## Network list
### Backbone(chronological order)
Common Type:
- [x] AlexNet                 2012 
- [x] VGG                     2014
- [x] Inception/GoogLeNet     2014
- [x] InceptionV2/V3          2015
- [ ] InceptionV4/Res         2016
- [x] ResNet                  2015
- [ ] ResNeXt                 2016
- [x] DenseNet                2016
- [ ] NasNet                  2017
- [ ] SeNet                   2017
- [ ] EfficientNet            2019

Light Type:
- [ ] ShuffleNet              2017
- [ ] MobileNet               2017

Other:
- [x] BasicGNN

### Pose Estimation(chronological order)
- [x] OpenPose                2015
- [x] Hourglass               2015
- [x] GNNlikeCNN              2015
- [x] SimpleBaseline          2017
- [ ] CPN                     2017
- [x] OpenPose                2017
- [x] FPNPoseNet              2017
- [ ] HRNet                   2018
- [x] CPN+GNN                 2018
- [ ] Multi-CPN               2019

### Segmentation(chronological order)
- [x] FCN                     2015

### Detection(chronological order)
- [x] YOLO
- [ ] RCNN 
- [ ] Fast-RCNN
- [ ] Faster-RCNN
- [ ] Mask-RCNN
- [ ] Mask-Score-RCNN

## Future Work: Pytorch loss
Will release a new repertory in the future
- [ ] Centerloss              2015
- [x] Focusloss               2016 
- [x] labelsmooth
...

