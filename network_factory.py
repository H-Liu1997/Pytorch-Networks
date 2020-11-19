from backbone import ResNet2015
from backbone import RegNet2020
from backbone import effnet

NET_LUT = {
        'resnet': ResNet2015.ResNet,
        'regnet': RegNet2020.RegNet,
        'resnext': RegNet2020.AnyNet,
        'effnet': effnet.EffNet,
    }

def load_regnet_weight(model,pretrain_path,sub_name):
    from collections import OrderedDict
    import torch
    checkpoints = torch.load(pretrain_path+WEIGHT_LUT[sub_name])
    states_no_module = OrderedDict()
    for k, v in checkpoints['model_state'].items():
        if k != 'head.fc.weight' and k!= 'head.fc.bias' and k!= 'stem.conv.weight':
            #print(k)
            name_no_module = k
            states_no_module[name_no_module] = v
    model.load_state_dict(states_no_module,strict=False)


LOAD_LUT = {
        'resnet': ResNet2015.ResNet,
        'regnet': load_regnet_weight,
        'resnext': load_regnet_weight,
        'effnet': load_regnet_weight,
    }

WEIGHT_LUT = {
        'RegNetY-8.0GF': 'regnet/RegNetY-8.0GF_dds_8gpu.pyth',
        'RegNetX-4.0GF': 'regnet/RegNetX-4.0GF_dds_8gpu.pyth',
        'RegNetY-3.2GF': 'regnet/RegNetY-3.2GF_dds_8gpu.pyth',
        'RegNetY-32GF': 'regnet/RegNetY-32GF_dds_8gpu.pyth',
        'ResNeXt-50': 'resnext/X-50-32x4d_dds_8gpu.pyth',
        'EfficientNet-B2': 'effnet/EN-B2_dds_8gpu.pyth',
    }


def get_network(net_name, logger=None, cfg=None):
    try:
        net_class = NET_LUT.get(net_name)
    except:
        logger.error("network tpye error, {} not exist".format(net_name))
    net_instance = net_class(cfg=cfg, logger=logger)
    if cfg.PRETRAIN is not None:
        load_func = LOAD_LUT.get(net_name)
        load_func(net_instance,cfg.PRETRAIN_PATH,cfg.PRETRAIN)
        logger.info("load {} pretrain weight success".format(net_name))
    return net_instance
    

if __name__ == "__main__":
    import logging
    from config_lyft import cfg
    from utils import load_cfg
    from ptflops import get_model_complexity_info
    import torch

    logger = load_cfg(cfg) 
    model = get_network('regnet', logger=logger, cfg=cfg.MODEL)
    model = model.cuda() if torch.cuda.is_available() else model
    flops, params = get_model_complexity_info(model,  (25, 224, 224), 
        as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
