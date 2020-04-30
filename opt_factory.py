import torch


def get_opt(model, cfg_train, logger=None):
    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    if cfg_train.TYPE == 'adam':
        opt = torch.optim.Adam(trainable_vars, 
            lr=cfg_train.LR, 
            betas=(cfg_train.BETA1, cfg_train.BETA2),
            eps=1e-08, 
            weight_decay=cfg_train.WEIGHT_DECAY,
            amsgrad=False)
    elif cfg_train.TYPE == 'sgd':
        opt = torch.optim.SGD(trainable_vars, 
            lr=cfg_train.LR, 
            momentum=cfg_train.BETA1,
            weight_decay=cfg_train.WEIGHT_DECAY,)
    else:
        logger.error("{} not exist in opt type".format(cfg_train.TYPE)) 

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,cfg_train.LR_REDUCE,
                                                       gamma=cfg_train.LR_FACTOR,
                                                       last_epoch=-1)
    return opt,lr_scheduler