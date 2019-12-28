# --------------------------------------------------------------------------- #
# Network in Network, ICML2014, https://arxiv.org/abs/1312.4400
# pytorch implementation by Haiyang Liu (haiyangliu1997@gmail.com)
# --------------------------------------------------------------------------- #


import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['NIN']


class NIN(nn.Module):
    def __init__(self):
        super(NIN,self).__init__()
        self.net_layers=nn.Sequential(
            nn.Conv2d(3,96,11,4,0),
            nn.ReLU(),
            nn.Conv2d(96,96,1,1,0),
            nn.ReLU(),
            nn.Conv2d(96,96,1,1,0),
            nn.ReLU(),

            nn.Conv2d(96,256,5,1,2),
            nn.ReLU(),
            nn.Conv2d(256,256,1,1,0),
            nn.ReLU(),
            nn.Conv2d(256,256,1,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(256,384,3,1,1),
            nn.ReLU(),
            nn.Conv2d(384,384,1,1,0),
            nn.ReLU(),
            nn.Conv2d(384,384,1,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.Conv2d(384,1024,3,1,1),
            nn.ReLU(),
            nn.Conv2d(1024,1024,1,1,0),
            nn.ReLU(),
            nn.Conv2d(1024,1024,1,1,0),
            nn.ReLU(),
            nn.MaxPool2d(2,2),

            nn.AdaptiveAvgPool2d((1,1)),

            nn.Softmax(dim=1),   
        )
        self.initialization()
    
    def initialization(self):
        _flag = 1 
        for layer in self.net_layers:
            if isinstance(layer,nn.Conv2d) or isinstance(layer,nn.Linear):
                nn.init.normal_(layer.weight, mean=0.0, std=0.01)
                print('init layer:',_flag,' weight success')
                if _flag == 1 or _flag == 3:
                    nn.init.constant_(layer.bias, 0.0)
                    print('init layer:',_flag,'bias to 0')
                else:
                    nn.init.constant_(layer.bias, 1.0)
                    print('init layer:',_flag,'bias to 1')
                _flag += 1
                
    def forward(self,input_):
        x = self.net_layers(input_)
        return x       


def _test():
    from torchsummary import summary
    model = NIN()
    model = model.cuda()
    summary(model,input_size=(3,224,224))
    

if __name__ == "__main__":
    _test()


# ------------------------------- model summary ----------------------------- #
#           Conv2d-21         [-1, 1024, 13, 13]       3,539,968
#              ReLU-22         [-1, 1024, 13, 13]               0
#            Conv2d-23         [-1, 1024, 13, 13]       1,049,600
#              ReLU-24         [-1, 1024, 13, 13]               0
#            Conv2d-25         [-1, 1024, 13, 13]       1,049,600
#              ReLU-26         [-1, 1024, 13, 13]               0
#         MaxPool2d-27           [-1, 1024, 6, 6]               0
# AdaptiveAvgPool2d-28           [-1, 1024, 1, 1]               0
#           Softmax-29           [-1, 1024, 1, 1]               0
# ================================================================
# Total params: 7,619,776
# Trainable params: 7,619,776
# Non-trainable params: 0
# ----------------------------------------------------------------
# Input size (MB): 0.57
# Forward/backward pass size (MB): 69.94
# Params size (MB): 29.07
# Estimated Total Size (MB): 99.58
# ---------------------------------- end ------------------------------------ #