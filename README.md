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
- [x] AlexNet2012 
- [x] VGG2014
- [ ] Inception/GoogLeNet2014
- [x] ResNet2015
- [ ] ResNeXt2016
- [x] DenseNet2016
- [ ] NasNet2017
- [ ] SeNet2017
- [ ] EfficientNet2019

Light Type:
- [ ] ShuffleNet2017
- [ ] MobileNet2017

Other:
- [x] BasicGNN

### Pose Estimation(chronological order)
- [x] OpenPose2015
- [x] Hourglass2015
- [x] GNNlikeCNN2015
- [x] SimpleBaseline2017
- [ ] CPN2017
- [x] OpenPose2017
- [x] FPNPoseNet2017
- [ ] HRNet2018
- [x] CPN+GNN2018
- [ ] Multi-CPN2019

### Segmentation(chronological order)
- [x] FCN2015

### Detection(chronological order)
- [x] YOLO
- [ ] RCNN 
- [ ] Fast-RCNN
- [ ] Faster-RCNN
- [ ] Mask-RCNN
- [ ] Mask-Score-RCNN

## Future Work: Pytorch loss
Will release a new repertory in the future
- [ ] Centerloss2015
- [x] Focusloss2016 
- [x] labelsmooth
...

