import torch.nn as nn
import torch.nn.functional as F
import torch


class BCE_Only_Loss2G(nn.Module):
    def __init__(self, cfg=None):
        super(BCE_Only_Loss2G, self).__init__()
       
    def forward(self, fake_outputs, real_target):
        final_loss = F.cross_entropy(fake_outputs, real_target)
        return final_loss

LOSS_FUNC_LUT = {
        'bce_only_g': BCE_Only_Loss2G,
    }


def get_loss_func(loss_name, **kwargs):    
    try:
        loss_func_class = LOSS_FUNC_LUT.get(loss_name)   
    except:
        kwargs['logger'].error("loss tpye error, {} not exist".format(loss_name))
    if 'scaling' in kwargs:
        loss_func = loss_func_class(kwargs['scaling'])
    else:
        loss_func = loss_func_class()
    return loss_func