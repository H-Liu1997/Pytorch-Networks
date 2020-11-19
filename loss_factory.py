import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np


class BCE_Only_Loss2G(nn.Module):
    def __init__(self, cfg=None):
        super(BCE_Only_Loss2G, self).__init__()
       
    def forward(self, fake_outputs, real_target):
        final_loss = F.cross_entropy(fake_outputs, real_target)
        return final_loss


class pytorch_neg_multi_log_likelihood_batch(nn.Module):
    def __init__(self, cfg=None):
        super(pytorch_neg_multi_log_likelihood_batch, self).__init__()
    
    def forward(self, gt, pred, confidences, avails):
        # convert to (batch_size, num_modes, future_len, num_coords)
        gt = torch.unsqueeze(gt, 1)  # add modes
        avails = avails[:, None, :, None]  # add modes and cords

        # error (batch_size, num_modes, future_len)
        error = torch.sum(((gt - pred) * avails) ** 2, dim=-1)  # reduce coords and use availability

        with np.errstate(divide="ignore"):  # when confidence is 0 log goes to -inf, but we're fine with it
            # error (batch_size, num_modes)
            error = torch.log(confidences) - 0.5 * torch.sum(error, dim=-1)  # reduce time

        # use max aggregator on modes for numerical stability
        # error (batch_size, num_modes)
        max_value, _ = error.max(dim=1, keepdim=True)  # error are negative at this point, so max() gives the minimum one
        error = -torch.log(torch.sum(torch.exp(error - max_value), dim=-1, keepdim=True)) - max_value  # reduce modes
        # print("error", error)
        return torch.mean(error)


LOSS_FUNC_LUT = {
        'nll_loss': pytorch_neg_multi_log_likelihood_batch,
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


