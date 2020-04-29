
from collections import OrderedDict


class SelfData(object):
    def __init__(self,value):
        self.value = value
        self.counter = 0+1e-8

    def add_value(self,add_value):
        self.counter += 1
        self.value += add_value.data.cpu().numpy()

    def avg(self):
        return self.value/self.counter


class CalculateAcc(object):
    def __init__(self):
        self.count_success_a = 0 + 1e-8
        self.count = 0+ 1e-8
        
    def add_value(self,output,target):
        self.count += output.shape[0]

        pred = output.data.max(1,keepdim=True)[1].cpu()
        
        self.count_success_a += pred.eq(target.data.view_as(pred)).sum().cpu().numpy()


    def print_(self):
        return (self.count_success_a/self.count)


def load_test_checkpoints(model, save_path, logger):
    try:
        #logger.debug(save_path.EXPS+save_path.NAME+save_path.GMODEL)
        states= torch.load(save_path.EXPS+save_path.NAME+save_path.MODEL) if torch.cuda.is_available() \
            else torch.load(save_path.EXPS+save_path.NAME+save_path.MODEL, map_location=torch.device('cpu'))
        #logger.debug("success")
        try:
            model.load_state_dict(states['model_state'])
        except:
            states_no_module = OrderedDict()
            for k, v in states['model_state'].items():
                name_no_module = k[7:]
                states_no_module[name_no_module] = v
            model.load_state_dict(states_no_module)
        logger.info('loading checkpoints success')
    except:
        logger.error("no checkpoints")


