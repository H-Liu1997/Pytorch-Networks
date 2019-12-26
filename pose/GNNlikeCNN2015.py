# ------------------------------------------------------------------------------
# Gnn like cnn Pytorch Implementation
# paper:
# https://arxiv.org/abs/1603.09065
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

# import torch_geometric.transforms as T
# from torch_geometric.nn import GCNConv,GatedGraphConv
# from torch_geometric.data import Data

device = torch.device('cuda')
edge_matrix = [[0,1,14,15],[1,0,2,5],[2,1,3,4,8],[3,2,4],[4,2,3],
                            [5,1,6,7,11],[6,5,7],[7,5,6],[8,2,9,10],[9,8,10],[10,8,9],
                            [11,5,12,13],[12,11,13],[13,11,12],[14,0,15,16],[15,0,14,17],[16,14],[17,15]]
#edge_matrix = edge_matrix.to(device)
edge_matrix_paf = [[1,8],[8,9],[9,10],[1,11],[11,12],[12,13],
                            [1,2],[2,3],[3,4],[2,16],[1,5],[5,6],[6,7],
                            [5,17],[0,1],[0,14],[0,15],[14,16],[15,17]]
#edge_matrix_paf = edge_matrix_paf.to(device)

def get_input(i,gnn_input):
    length = len(edge_matrix[i])
    select_index = torch.tensor(edge_matrix[i])
    select_index = select_index.to(device)
    #print(select_index)
    input_new = torch.index_select(gnn_input,1,select_index)
    return input_new,length

def get_input_paf(i,gnn_input):
    select_index = torch.tensor(edge_matrix_paf[i])
    select_index = select_index.to(device)
    select_index_1 = torch.tensor([2*i+19,2*i+20])
    select_index_1 = select_index_1.to(device)
    input_paf = torch.index_select(gnn_input,1,select_index_1)
    #print(input_paf.size())
    input_new = torch.index_select(gnn_input,1,select_index)
    #print(input_new.size())
    input_final = torch.cat([input_new,input_paf],1)
    #print(input_final.size())
    return input_final


class Model_GNN(nn.Module):
    ''' A GGNN module, input 19 nodes,
    CNN input size: N * 19 * H * W 
    '''
    def __init__(self, Gnn_layers, use_gpu):
        super().__init__()
        self.gnn_0 = nn.ModuleList([nn.Conv2d(in_channels = 4, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_1 = nn.ModuleList([nn.Conv2d(in_channels = 4, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])
            
        self.gnn_2 = nn.ModuleList([nn.Conv2d(in_channels = 5, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_3 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])
        
        self.gnn_4 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])
        
        self.gnn_5 = nn.ModuleList([nn.Conv2d(in_channels = 5, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])
        
        self.gnn_6 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_7 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        
        self.gnn_8 = nn.ModuleList([nn.Conv2d(in_channels = 4, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_9 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_10 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_11 = nn.ModuleList([nn.Conv2d(in_channels = 4, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])


        self.gnn_12 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_13 = nn.ModuleList([nn.Conv2d(in_channels = 3, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_14 = nn.ModuleList([nn.Conv2d(in_channels = 4, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_15 = nn.ModuleList([nn.Conv2d(in_channels = 4, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_16 = nn.ModuleList([nn.Conv2d(in_channels = 2, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)])

        self.gnn_17 = nn.ModuleList([nn.Conv2d(in_channels = 2, out_channels = 64,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 64, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   #nn.Conv2d(in_channels = 128, out_channels = 128,
                                   #kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 128,
                                   kernel_size = (7,7), stride = 1,padding = 3),
                                   nn.Conv2d(in_channels = 128, out_channels = 512,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                   nn.Conv2d(in_channels = 512, out_channels = 1,
                                   kernel_size = (1,1), stride = 1,padding = 0)]) 

        self.gnn_actfs = nn.ModuleList([nn.ReLU() for l in range(7)])
        self.use_gpu = use_gpu
        self._initialize_weights_norm()

    def forward(self, out1, cnn_output, gnn_interations):
        N = cnn_output.size()[0]
        C = cnn_output.size()[1]
        H = cnn_output.size()[2]
        W = cnn_output.size()[3]
       
        gnn_output = cnn_output.clone()
        gnn_output_final = cnn_output.clone()
        
        for n in range(18):# for n samples
            gnn_out_one_batch,length = get_input(n, gnn_output)
            #gnn_out_one_batch = torch.cat([gnn_out_one_batch,out1],1)

            if n == 1:
                for idx, g_layer in enumerate(self.gnn_1):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 2:
                for idx, g_layer in enumerate(self.gnn_2):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 3:
                for idx, g_layer in enumerate(self.gnn_3):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 0:
                for idx, g_layer in enumerate(self.gnn_0):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 4:
                for idx, g_layer in enumerate(self.gnn_4):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            elif n == 5:
                for idx, g_layer in enumerate(self.gnn_5):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 6:
                for idx, g_layer in enumerate(self.gnn_6):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            elif n == 7:
                for idx, g_layer in enumerate(self.gnn_7):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 8:
                for idx, g_layer in enumerate(self.gnn_8):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            elif n == 9:
                for idx, g_layer in enumerate(self.gnn_9):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 10:
                for idx, g_layer in enumerate(self.gnn_10):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            elif n == 11:
                for idx, g_layer in enumerate(self.gnn_11):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 12:
                for idx, g_layer in enumerate(self.gnn_12):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            elif n == 13:
                for idx, g_layer in enumerate(self.gnn_13):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 14:
                for idx, g_layer in enumerate(self.gnn_14):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            elif n == 15:
                for idx, g_layer in enumerate(self.gnn_15):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            elif n == 16:
                for idx, g_layer in enumerate(self.gnn_16):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            else:
                for idx, g_layer in enumerate(self.gnn_17):
                    if idx == 6:
                        gnn_out_one_batch = g_layer(gnn_out_one_batch)
                    else:
                        gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))
            #print(gnn_out_one_batch.size())
            gnn_output_final[:,n:n+1,:,:] = gnn_out_one_batch
        #print(gnn_output_final.size())
        return gnn_output_final 
    
    def _initialize_weights_norm(self):

        # last layer of these block don't have Relu
        for i in range(5):
            init.zeros_(self.gnn_0[i].weight)
            init.zeros_(self.gnn_1[i].weight)
            init.zeros_(self.gnn_2[i].weight)
            init.zeros_(self.gnn_3[i].weight)
            init.zeros_(self.gnn_4[i].weight)
            init.zeros_(self.gnn_5[i].weight)
            init.zeros_(self.gnn_6[i].weight)
            init.zeros_(self.gnn_7[i].weight)
            init.zeros_(self.gnn_8[i].weight)

            init.zeros_(self.gnn_9[i].weight)
            init.zeros_(self.gnn_10[i].weight)
            
            init.zeros_(self.gnn_11[i].weight)
            init.zeros_(self.gnn_12[i].weight)
            init.zeros_(self.gnn_13[i].weight)
            init.zeros_(self.gnn_14[i].weight)
            init.zeros_(self.gnn_15[i].weight)
            init.zeros_(self.gnn_16[i].weight)
            init.zeros_(self.gnn_17[i].weight)

class Model_GNN_paf(nn.Module):
    ''' A GGNN module, input 19 nodes,
    CNN input size: N * 38 * H * W 
    '''
    def __init__(self, Gnn_layers, use_gpu):
        super().__init__()
        self.gnn_layers = nn.ModuleList([nn.Conv2d(in_channels = 4, out_channels = 16,
                                   kernel_size = (5,5), stride = 1,padding = 2),
                                         nn.Conv2d(in_channels = 16, out_channels = 16,
                                   kernel_size = (5,5), stride = 1,padding = 2),
                                   nn.Conv2d(in_channels = 16, out_channels = 32,
                                   kernel_size = (5,5), stride = 1,padding = 2),
                                   nn.Conv2d(in_channels = 32, out_channels = 32,
                                   kernel_size = (5,5), stride = 1,padding = 2),
                                   nn.Conv2d(in_channels = 32, out_channels = 32,
                                   kernel_size = (5,5), stride = 1,padding = 2),
                                         nn.Conv2d(in_channels = 32, out_channels = 128,
                                   kernel_size = (1,1), stride = 1,padding = 0),
                                         nn.Conv2d(in_channels = 128, out_channels = 2,
                                   kernel_size = (1,1), stride = 1,padding = 0)])
        self.gnn_actfs = nn.ModuleList([nn.ReLU() for l in range(7)])
        self.use_gpu = use_gpu
        self._initialize_weights_norm()

    def forward(self, cnn_output, gnn_interations):
        N = cnn_output.size()[0]
        C = 38
        H = cnn_output.size()[2]
        W = cnn_output.size()[3]
        gnn_output = cnn_output.clone()
        
        index_s = torch.tensor([0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,
                                24,25,26,27,28,29,30,31,32,33,34,35,36,37])
        index_s = index_s.to(device)
        gnn_output_final = torch.index_select(gnn_output,1,index_s)

        for n in range(19):# for n samples
            gnn_out_one_batch = get_input_paf(n, gnn_output)
            gnn_out_one_batch.to(device)
            #print('pafin',gnn_out_one_batch.size())

            for idx, g_layer in enumerate(self.gnn_layers):
                if idx == 6:
                    gnn_out_one_batch = g_layer(gnn_out_one_batch)
                else:
                    gnn_out_one_batch = self.gnn_actfs[idx](g_layer(gnn_out_one_batch))

            gnn_output_final[:,2*n: 2*n + 1,:,:] = gnn_out_one_batch[:,0:1,:,:]
            gnn_output_final[:,2*n + 1:2*n+2,:,:] = gnn_out_one_batch[:,1:2,:,:]
        #print('paf',gnn_output_final.size())
        return gnn_output_final 

    def _initialize_weights_norm(self):

        # last layer of these block don't have Relu
        for i in range(7):
            init.normal_(self.gnn_layers[i].weight, std=0.01)
               



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

        return out1,(out6_1, out6_2), saved_for_loss

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
        out1,x_loss,saved_for_loss = self.cnn.forward(input)
        x_heatmap = x_loss[1].clone()
        x_paf = x_loss[0].clone() 
        x_paf_input = torch.cat([x_heatmap,x_paf],1)

        if use_gnn:
            y = self.gnn.forward(out1,x_heatmap,gnn_interations)
            out7_2 =  y + x_heatmap 
            z = self.gnn_paf.forward(x_paf_input,gnn_interations) 
            out7_1 =  x_paf 
            saved_for_loss.append(out7_1)
            saved_for_loss.append(out7_2)
            x_loss_gnn = (out7_1,out7_2)
            return x_loss_gnn,saved_for_loss
        else:
            return x_loss,saved_for_loss 



        

 
