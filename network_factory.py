from backbone import ResNet2015

NET_LUT = {
        'resnet': ResNet2015.ResNet,
    }

def get_network(net_name, logger=None, cfg=None):
    try:
        net_class = NET_LUT.get(net_name)
    except:
        logger.error("network tpye error, {} not exist".format(net_name))
    net_instance = net_class(cfg=cfg, logger=logger)
    return net_instance
    

if __name__ == "__main__":
    import logging
    from config import cfg, load_cfg
    from ptflops import get_model_complexity_info
    logger = load_cfg() 
    model = get_network('resnet', logger=logger, cfg=cfg.MODEL)
    model = model.cuda()
    flops, params = get_model_complexity_info(model,  (3, 224, 224), 
        as_strings=True, print_per_layer_stat=True)
    print('{:<30}  {:<8}'.format('Computational complexity: ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters: ', params))
