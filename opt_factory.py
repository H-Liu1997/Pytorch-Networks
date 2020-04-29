import torch


def get_opt(model, cfg_train, logger=None):
    
    if cfg_train.TYPE == 'adam':
        trainable_vars = [param for param in model.parameters() if param.requires_grad]
        opt = torch.optim.Adam(trainable_vars, 
            lr=cfg_train.LR, 
            betas=(cfg_train.BETA1, cfg_train.BETA2),
            eps=1e-08, 
            weight_decay=cfg_train.WEIGHT_DECAY,
            amsgrad=False)
    else:
        logger.error("{} not exist in opt type".format(cfg_train.TYPE)) 
    return opt