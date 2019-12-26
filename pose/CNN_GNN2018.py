# ------------------------------------------------------------------------------
# CPM+GNN Pytorch Implementation
# CPN+GNN paper:
# https://arxiv.org/abs/1901.01760
# Written by Haiyang Liu (haiyangliu1997@gmail.com)
# ------------------------------------------------------------------------------
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torch.utils.model_zoo as model_zoo
from torch.autograd import Variable
from torch.nn import init

import torch_geometric.transforms as T
from torch_geometric.nn import GCNConv,GatedGraphConv
from torch_geometric.data import Data

edge_ = torch.tensor([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],
                            [3,4], [4,3],[2,4],[4,2],[1,5],[5,1],
                            [5,6],[6,5],[6,7],[7,6],[5,7],[7,5],
                            [1,8],[8,1],[8,9],[9,8],[9,10],[10,9],
                            [8,10],[10,8],[1,11],[11,1],[11,12],[12,11],
                            [12,13],[13,12],[11,13],[13,11],[0,14],[14,0],
                            [14,16],[16,14],[0,15],[15,0],[15,17],
                            [17,15]],dtype = torch.long)
edge_index = edge_.t().contiguous()   
device = torch.device('cuda')
edge_index = edge_index.to(device) 
# paf connection
edge_paf = torch.tensor([[0,2],[2,0],[1,3],[3,1],
                         [2,4],[4,2],[3,5],[5,3],
                         [6,8],[8,6],[7,9],[9,7],
                         [8,10],[10,8],[9,11],[11,9],
                         [0,12],[12,0],[1,13],[13,1],
                         [12,28],[28,12],[13,29],[29,13],
                         [20,6],[6,20],[21,7],[7,21],
                         [20,28],[28,20],[21,29],[29,21],
                         [32,28],[28,32],[29,33],[33,29],
                         [30,28],[28,30],[29,31],[31,29],
                         [34,30],[30,34],[31,35],[35,31],
                         [36,32],[32,36],[33,37],[37,33],
                         [36,26],[26,36],[37,27],[27,37],
                         [20,26],[26,20],[21,27],[27,21],
                         [14,16],[16,14],[15,17],[17,15],
                         [22,24],[24,22],[23,25],[25,23],
                         [18,12],[18,12],[19,13],[13,19],
                         [34,18],[18,34],[19,35],[35,19],
                         [20,22],[22,21],[21,23],[23,21],
                         [12,14],[14,12],[13,15],[15,13]    
                            ],dtype = torch.long)
edge_index_paf = edge_paf.t().contiguous()   
#device = torch.device('cuda')
edge_index_paf = edge_index_paf.to(device) 

'''
def data_transform(cnn_output_one_batch):
    edge_index = torch.tensor([[0,1],[1,0],[1,2],[2,1],[2,3],[3,2],
                            [3,4], [4,3],[2,4],[4,2],[1,5],[5,1],
                            [5,6],[6,5],[6,7],[7,6],[5,7],[7,5],
                            [1,8],[8,1],[8,9],[9,8],[9,10],[10,9],
                            [8,10],[10,8],[1,11],[11,1],[11,12],[12,11],
                            [12,13],[13,12],[11,13],[13,11],[0,14],[14,0],
                            [14,16],[16,14],[0,15],[15,0],[15,17],
                            [17,15]],dtype = torch.long)

    data = Data(x = cnn_output_one_batch, edge_index = edge_index.t().contiguous())
    device = torch.device('cuda')
    data = data.to(device)
    return data'''

class Model_GNN(nn.Module):
    ''' A GGNN module, input 19 nodes,
    CNN input size: N * 19 * H * W 
    '''
    def __init__(self, Gnn_layers, use_gpu):
        super().__init__()
        self.gnn_layers = nn.ModuleList([GatedGraphConv(2116, 2) for l in range(Gnn_layers)])
        #self.gnn_layers = nn.ModuleList([GCNConv(2116,2116) for l in range(Gnn_layers)])
        #self.gnn_actfs = nn.ModuleList([nn.LeakyReLU() for l in range(Gnn_layers)])
        self.use_gpu = use_gpu

    def forward(self, cnn_output, gnn_interations):
        N = cnn_output.size()[0]
        C = cnn_output.size()[1]
        H = cnn_output.size()[2]
        W = cnn_output.size()[3]
        gnn_output = cnn_output.view(N,C,H*W) 
        gnn_output_1 = gnn_output.clone()
        #gnn_output_2 = gnn_output.clone()
        #gnn_output_3 = gnn_output.clone()
                   
        #for i in range(gnn_interations): #propagate time 
        
        for n in range(N):# for n samples
            #gnn_input_edge = data_transform(gnn_output[n]) # change to special data structure: data
            for idx, g_layer in enumerate(self.gnn_layers):
                #gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_output[n], edge_index))
                gnn_out_one_batch = g_layer(gnn_output[n], edge_index)
            gnn_output_1[n] = gnn_out_one_batch

        '''
        for n in range(N):# for n samples
            #gnn_input_edge = data_transform(gnn_output[n]) # change to special data structure: data
            for idx, g_layer in enumerate(self.gnn_layers):
                gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_output_1[n], edge_index))    
            gnn_output_2[n] = gnn_out_one_batch   ''' 

        gnn_final = gnn_output_1.view(N,C,H,W)        
        return gnn_final 

class Model_GNN_paf(nn.Module):
    ''' A GGNN module, input 19 nodes,
    CNN input size: N * 38 * H * W 
    '''
    def __init__(self, Gnn_layers, use_gpu):
        super().__init__()
        self.gnn_layers = nn.ModuleList([GatedGraphConv(2116, 2) for l in range(Gnn_layers)])
        #self.gnn_layers = nn.ModuleList([GCNConv(2116,2116) for l in range(Gnn_layers)])
        #self.gnn_actfs = nn.ModuleList([nn.LeakyReLU() for l in range(Gnn_layers)])
        self.use_gpu = use_gpu

    def forward(self, cnn_output, gnn_interations):
        N = cnn_output.size()[0]
        C = cnn_output.size()[1]
        H = cnn_output.size()[2]
        W = cnn_output.size()[3]
        gnn_output = cnn_output.view(N,C,H*W) 
        gnn_output_1 = gnn_output.clone()
        #gnn_output_2 = gnn_output.clone()
        #gnn_output_3 = gnn_output.clone()
                   
        #for i in range(gnn_interations): #propagate time 
        
        for n in range(N):# for n samples
            #gnn_input_edge = data_transform(gnn_output[n]) # change to special data structure: data
            for idx, g_layer in enumerate(self.gnn_layers):
                #gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_output[n], edge_index_paf))
                gnn_out_one_batch = g_layer(gnn_output[n], edge_index_paf)
            gnn_output_1[n] = gnn_out_one_batch

        '''
        for n in range(N):# for n samples
            #gnn_input_edge = data_transform(gnn_output[n]) # change to special data structure: data
            for idx, g_layer in enumerate(self.gnn_layers):
                gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_output_1[n], edge_index))    
            gnn_output_2[n] = gnn_out_one_batch   ''' 

        gnn_final = gnn_output_1.view(N,C,H,W)        
        return gnn_final               

# some founction to build cnn
def make_stages(cfg_dict):
    """Builds CPM stages from a dictionary
    Args:
        cfg_dict: a dictionary
    """
    layers = []
    for i in range(len(cfg_dict) - 1):
        one_ = cfg_dict[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    one_ = list(cfg_dict[-1].keys())
    k = one_[0]
    v = cfg_dict[-1][k]
    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                       kernel_size=v[2], stride=v[3], padding=v[4])
    layers += [conv2d]
    return nn.Sequential(*layers)

def make_vgg19_block(block):
    """Builds a vgg19 block from a dictionary
    Args:
        block: a dictionary
    """
    layers = []
    for i in range(len(block)):
        one_ = block[i]
        for k, v in one_.items():
            if 'pool' in k:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1],
                                        padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1],
                                   kernel_size=v[2], stride=v[3],
                                   padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)

def get_model(trunk='vgg19'):
   
    blocks = {}
    # block0 is the preprocessing stage
    if trunk == 'vgg19':
        block0 = [{'conv1_1': [3, 64, 3, 1, 1]},
                  {'conv1_2': [64, 64, 3, 1, 1]},
                  {'pool1_stage1': [2, 2, 0]},
                  {'conv2_1': [64, 128, 3, 1, 1]},
                  {'conv2_2': [128, 128, 3, 1, 1]},
                  {'pool2_stage1': [2, 2, 0]},
                  {'conv3_1': [128, 256, 3, 1, 1]},
                  {'conv3_2': [256, 256, 3, 1, 1]},
                  {'conv3_3': [256, 256, 3, 1, 1]},
                  {'conv3_4': [256, 256, 3, 1, 1]},
                  {'pool3_stage1': [2, 2, 0]},
                  {'conv4_1': [256, 512, 3, 1, 1]},
                  {'conv4_2': [512, 512, 3, 1, 1]},
                  {'conv4_3_CPM': [512, 256, 3, 1, 1]},
                  {'conv4_4_CPM': [256, 128, 3, 1, 1]}]

    elif trunk == 'mobilenet':
        block0 = [{'conv_bn': [3, 32, 2]},  # out: 3, 32, 184, 184
                  {'conv_dw1': [32, 64, 1]},  # out: 32, 64, 184, 184
                  {'conv_dw2': [64, 128, 2]},  # out: 64, 128, 92, 92
                  {'conv_dw3': [128, 128, 1]},  # out: 128, 256, 92, 92
                  {'conv_dw4': [128, 256, 2]},  # out: 256, 256, 46, 46
                  {'conv4_3_CPM': [256, 256, 1, 3, 1]},
                  {'conv4_4_CPM': [256, 128, 1, 3, 1]}]

    # Stage 1
    blocks['block1_1'] = [{'conv5_1_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L1': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L1': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L1': [512, 38, 1, 1, 0]}]

    blocks['block1_2'] = [{'conv5_1_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_2_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_3_CPM_L2': [128, 128, 3, 1, 1]},
                          {'conv5_4_CPM_L2': [128, 512, 1, 1, 0]},
                          {'conv5_5_CPM_L2': [512, 19, 1, 1, 0]}]

    # Stages 2 - 6
    for i in range(2, 7):
        blocks['block%d_1' % i] = [
            {'Mconv1_stage%d_L1' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L1' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L1' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L1' % i: [128, 38, 1, 1, 0]}
        ]

        blocks['block%d_2' % i] = [
            {'Mconv1_stage%d_L2' % i: [185, 128, 7, 1, 3]},
            {'Mconv2_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv3_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv4_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv5_stage%d_L2' % i: [128, 128, 7, 1, 3]},
            {'Mconv6_stage%d_L2' % i: [128, 128, 1, 1, 0]},
            {'Mconv7_stage%d_L2' % i: [128, 19, 1, 1, 0]}
        ]

    models_dict = {}

    if trunk == 'vgg19':
        print("Bulding VGG19")
        models_dict['block0'] = make_vgg19_block(block0)

    for k, v in blocks.items():
        models_dict[k] = make_stages(list(v))
    
    return models_dict

def use_vgg(model, model_path, trunk,weight_path):
    try:
        old_weights = torch.load(weight_path)
        vgg_keys = old_weights.keys()

        weights_load = {}
        # weight+bias,weight+bias.....(repeat 10 times)
        for i in range(len(vgg_keys)):
            weights_load[list(model.state_dict().keys())[i]
                        ] = old_weights[list(vgg_keys)[i]]

        state = model.state_dict()
        state.update(weights_load)
        model.load_state_dict(state)
        #model.load_state_dict(old_weights)
        print('success load old weights and epoch num:')
    except:
        model_urls = {
            'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
            'ssd': 'https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth',
            'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'}

        number_weight = {
            'vgg16': 18,
            'ssd': 18,
            'vgg19': 20}

        url = model_urls[trunk]

        if trunk == 'ssd':
            urllib.urlretrieve('https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth',
                            os.path.join(model_path, 'ssd.pth'))
            vgg_state_dict = torch.load(os.path.join(model_path, 'ssd.pth'))
            print('loading SSD')
        else:
            vgg_state_dict = model_zoo.load_url(url, model_dir=model_path)
        vgg_keys = vgg_state_dict.keys()

        # load weights of vgg
        weights_load = {}
        # weight+bias,weight+bias.....(repeat 10 times)
        for i in range(number_weight[trunk]):
            weights_load[list(model.state_dict().keys())[i]
                        ] = vgg_state_dict[list(vgg_keys)[i]]

        state = model.state_dict()
        state.update(weights_load)
        model.load_state_dict(state)
        print('load imagenet pretrained model: {}'.format(model_path))



class Model_CNN(nn.Module):
    def __init__(self, model_dict):
        super(Model_CNN, self).__init__()
        self.model0 = model_dict['block0']
        self.model1_1 = model_dict['block1_1']
        self.model2_1 = model_dict['block2_1']
        self.model3_1 = model_dict['block3_1']
        self.model4_1 = model_dict['block4_1']
        self.model5_1 = model_dict['block5_1']
        self.model6_1 = model_dict['block6_1']

        self.model1_2 = model_dict['block1_2']
        self.model2_2 = model_dict['block2_2']
        self.model3_2 = model_dict['block3_2']
        self.model4_2 = model_dict['block4_2']
        self.model5_2 = model_dict['block5_2']
        self.model6_2 = model_dict['block6_2']

        self._initialize_weights_norm()

    def forward(self, x):

        saved_for_loss = []
        out1 = self.model0(x)

        out1_1 = self.model1_1(out1)
        out1_2 = self.model1_2(out1)
        out2 = torch.cat([out1_1, out1_2, out1], 1)
        saved_for_loss.append(out1_1)
        saved_for_loss.append(out1_2)

        out2_1 = self.model2_1(out2)
        out2_2 = self.model2_2(out2)
        out3 = torch.cat([out2_1, out2_2, out1], 1)
        saved_for_loss.append(out2_1)
        saved_for_loss.append(out2_2)

        out3_1 = self.model3_1(out3)
        out3_2 = self.model3_2(out3)
        out4 = torch.cat([out3_1, out3_2, out1], 1)
        saved_for_loss.append(out3_1)
        saved_for_loss.append(out3_2)

        out4_1 = self.model4_1(out4)
        out4_2 = self.model4_2(out4)
        out5 = torch.cat([out4_1, out4_2, out1], 1)
        saved_for_loss.append(out4_1)
        saved_for_loss.append(out4_2)

        out5_1 = self.model5_1(out5)
        out5_2 = self.model5_2(out5)
        out6 = torch.cat([out5_1, out5_2, out1], 1)
        saved_for_loss.append(out5_1)
        saved_for_loss.append(out5_2)

        out6_1 = self.model6_1(out6)
        out6_2 = self.model6_2(out6)
        saved_for_loss.append(out6_1)
        saved_for_loss.append(out6_2)

        return (out6_1, out6_2), saved_for_loss

    def _initialize_weights_norm(self):

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight, std=0.01)
                if m.bias is not None:  # mobilenet conv2d doesn't add bias
                    init.constant_(m.bias, 0.0)

        # last layer of these block don't have Relu
        init.normal_(self.model1_1[8].weight, std=0.01)
        init.normal_(self.model1_2[8].weight, std=0.01)

        init.normal_(self.model2_1[12].weight, std=0.01)
        init.normal_(self.model3_1[12].weight, std=0.01)
        init.normal_(self.model4_1[12].weight, std=0.01)
        init.normal_(self.model5_1[12].weight, std=0.01)
        init.normal_(self.model6_1[12].weight, std=0.01)

        init.normal_(self.model2_2[12].weight, std=0.01)
        init.normal_(self.model3_2[12].weight, std=0.01)
        init.normal_(self.model4_2[12].weight, std=0.01)
        init.normal_(self.model5_2[12].weight, std=0.01)
        init.normal_(self.model6_2[12].weight, std=0.01)   


class Model_Total(nn.Module):
    def __init__(self, model_dict, gnn_layers, use_gpu):
        super().__init__()
        
        self.cnn = Model_CNN(model_dict)
        self.gnn = Model_GNN(gnn_layers,use_gpu)
        self.gnn_paf = Model_GNN_paf(gnn_layers,use_gpu)
    
    def forward(self, input, gnn_interations, use_gnn):
        x_loss,saved_for_loss = self.cnn.forward(input)
        x_heatmap = x_loss[1].clone()
        x_paf = x_loss[0].clone() 

        if use_gnn:
            y = self.gnn.forward(x_heatmap,gnn_interations)
            out7_2 = x_loss[1] + y
            z = self.gnn_paf.forward(x_paf,gnn_interations)
            out7_1 = x_loss[0] + z 
            saved_for_loss.append(out7_1)
            saved_for_loss.append(out7_2)
            x_loss_gnn = (out7_1,out7_2)
            return x_loss_gnn,saved_for_loss
        else:
            return x_loss,saved_for_loss 



        

 
