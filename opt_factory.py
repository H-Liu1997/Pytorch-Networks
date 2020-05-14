import torch

class WarmLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        self.optimizer = optimizer

        if not isinstance(lr_lambda, list) and not isinstance(lr_lambda, tuple):
            self.lr_lambdas = [lr_lambda] * len(optimizer.param_groups)
        else:
            if len(lr_lambda) != len(optimizer.param_groups):
                raise ValueError("Expected {} lr_lambdas, but got {}".format(
                    len(optimizer.param_groups), len(lr_lambda)))
            self.lr_lambdas = list(lr_lambda)
        self.last_epoch = last_epoch
        super(WarmLR, self).__init__(optimizer, last_epoch)


    def state_dict(self):
        import types
        state_dict = {key: value for key, value in self.__dict__.items() if key not in ('optimizer', 'lr_lambdas')}
        state_dict['lr_lambdas'] = [None] * len(self.lr_lambdas)

        for idx, fn in enumerate(self.lr_lambdas):
            if not isinstance(fn, types.FunctionType):
                state_dict['lr_lambdas'][idx] = fn.__dict__.copy()

        return state_dict


    def load_state_dict(self, state_dict):
        lr_lambdas = state_dict.pop('lr_lambdas')
        self.__dict__.update(state_dict)
        state_dict['lr_lambdas'] = lr_lambdas

        for idx, fn in enumerate(lr_lambdas):
            if fn is not None:
                self.lr_lambdas[idx].__dict__.update(fn)
       

    def get_lr(self):
        if self.last_epoch > 0:
            return [group['lr'] + lmbda(self.last_epoch)
                    for lmbda, group in zip(self.lr_lambdas, self.optimizer.param_groups)]
        else:
            return list(self.base_lrs)


def get_opt(model, cfg_train, logger=None, is_warm=False, its_total=0):
    trainable_vars = [param for param in model.parameters() if param.requires_grad]
    if is_warm:
        opt = torch.optim.SGD(trainable_vars, 
            lr=cfg_train.LR_WARM, 
            momentum=cfg_train.BETA1,
            weight_decay=cfg_train.WEIGHT_DECAY,)
        factor = float((cfg_train.LR) / its_total)
        lmbda = lambda its: factor
        lr_scheduler = WarmLR(opt, lmbda, last_epoch=-1)
        return opt, lr_scheduler

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

    if cfg_train.LR_TYPE == 'cos':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, its_total, eta_min=0, last_epoch=-1)
    else:
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(opt,cfg_train.LR_REDUCE,
                                                        gamma=cfg_train.LR_FACTOR,
                                                        last_epoch=-1)
    return opt,lr_scheduler