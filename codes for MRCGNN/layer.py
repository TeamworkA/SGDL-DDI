import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv,RGCNConv
import numpy as np
import csv
import os
import random
from nodeformer import NodeFormer

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def set_random_seed(seed, deterministic=False):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
set_random_seed(1, deterministic=True)
def reset_parameters(w):
    stdv = 1. / math.sqrt(w.size(0))
    w.data.uniform_(-stdv, stdv)

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        
        
        # 针对每个节点的特征计算权重
        b, c = x.size()
        y = self.fc(x)
        return x * y


class Discriminator(nn.Module):
    def __init__(self, n_h):
        super(Discriminator, self).__init__()
        self.f_k = nn.Bilinear(32, 32, 1)

        for m in self.modules():
            self.weights_init(m)

    def weights_init(self, m):
        if isinstance(m, nn.Bilinear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, c, h_pl, h_mi, s_bias1=None, s_bias2=None):
        c_x = c.expand_as(h_pl)

        sc_1 = self.f_k(h_pl, c_x)
        sc_2 = self.f_k(h_mi, c_x)

        if s_bias1 is not None:
            sc_1 += s_bias1
        if s_bias2 is not None:
            sc_2 += s_bias2

        logits = torch.cat((sc_1, sc_2), 1)

        return logits


class AvgReadout(nn.Module):
    def __init__(self):
        super(AvgReadout, self).__init__()

    def forward(self, seq, msk=None):
        if msk is None:
            return torch.mean(seq, 0)
        else:
            msk = torch.unsqueeze(msk, -1)
            return torch.sum(seq * msk, 0) / torch.sum(msk)

class MLP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MLP, self).__init__()

        self.linear1 = nn.Linear(in_channels, 2 * out_channels)
        self.linear2 = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)

        return x


class SGDL(nn.Module):
    def __init__(self, feature, hidden1, hidden2, decoder1, dropout,zhongzi,brics_dim,num_class):
        super(SGDL, self).__init__()


        self.encoder_o1 = RGCNConv(feature, hidden1,num_relations=num_class)
        self.encoder_o2 = RGCNConv(hidden1 , hidden2,num_relations=num_class)

        self.brics_projection = nn.Linear(brics_dim, hidden1)

        self.nodeformer = NodeFormer(
            in_channels=hidden1, 
            hidden_channels=hidden1, 
            out_channels=hidden2, 
            num_layers=2, 
            num_heads=4, 
            use_edge_loss=True  # 开启边正则化 Loss
        )
        #改进一
        self.layer_gate = nn.Parameter(torch.tensor([0.3, 0.3, 0.4]))


        #改进二
        self.se_rgcn1 = SELayer(hidden1, reduction=8)
        self.se_rgcn2 = SELayer(hidden2, reduction=8)
        self.se_nf     = SELayer(hidden2, reduction=8)

        self.disc = Discriminator(hidden2 * 2)#hidden2

        self.dropout = dropout
        self.sigm = nn.Sigmoid()
        self.read = AvgReadout()

        self.input_dim = (hidden1 + hidden2  + feature) * 2
        
        self.mlp = nn.ModuleList([
            nn.Linear(self.input_dim, 256),  
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 128),
            nn.ELU(),
            nn.Dropout(p=0.1),
            nn.Linear(128, num_class)
        ])
        drug_list = []
        with open('data/drug_listxiao.csv', 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                drug_list.append(row[0])
        features = np.load('feature/drug_emb_feature0.npy')

        ids = np.load('feature/drug_emb_feature0_drug_ids.npy', allow_pickle=True)
        ids = ids.tolist()
        features1 = []
        for i in range(len(drug_list)):
            features1.append(features[ids.index(drug_list[i])])
        features1 = np.array(features1)

        # self.features1 = torch.from_numpy(features1).cuda()  
        self.features1 = torch.from_numpy(features1).to('cuda' if torch.cuda.is_available() else 'cpu')

    def MLP(self, vectors, layer):
            for i in range(layer):
                vectors = self.mlp[i](vectors)

            return vectors
    #def forward(self, data_o, data_s, data_a, idx):
    def forward(self, data_o,data_s,brics_feat, idx):


        #RGCN for DDI event graph and two corrupted graph
        x_o, adj ,e_type= data_o.x, data_o.edge_index,data_o.edge_type
        x_s = data_s.x
        e_type=torch.tensor(e_type,dtype=torch.int64)

        adjs = [adj]

        #R-GCN分支（正样本）
        x1_o = F.relu(self.encoder_o1(x_o, adj,e_type))
        x1_o = F.dropout(x1_o, self.dropout, training=self.training)
        x2_o = self.encoder_o2(x1_o, adj,e_type)

        # x_brics_proj = F.elu(self.brics_projection(brics_feat)) # [N, hidden1]
        
        #构造负样本 BRICS 特征 
        # idx_perm = torch.randperm(x_brics_proj.size(0))
        # x_brics_s_proj = x_brics_proj[idx_perm]

        # nf_out_o, nf_loss_o = self.nodeformer(x_brics_proj, adjs)
        # nf_out_s, _ = self.nodeformer(x_brics_s_proj, adjs) # 负样本流

        #  对比学习判别器 
        h_global = self.read(nf_out_o) 
        h_global = self.sigm(h_global)
        # ret_os = self.disc(h_global, nf_out_o, nf_out_s)

        # 特征融合 
        x1_o = self.se_rgcn1(x1_o)
        x2_o = self.se_rgcn2(x2_o)
        nf_out_o = self.se_nf(nf_out_o) # 现在的 nf_out_o 来自 BRICS

        gate = torch.softmax(self.layer_gate, dim=0)

        # nf_out_o, nf_loss_o = self.nodeformer(x_brics_proj, adjs)

        # nf_out_s, _ = self.nodeformer(x_brics_s_proj, adjs) # 负样本流

        h_global = self.read(nf_out_o) 
        h_global = self.sigm(h_global)
        # ret_os = self.disc(h_global, nf_out_o, nf_out_s)

        x1_o = self.se_rgcn1(x1_o)
        x2_o = self.se_rgcn2(x2_o)
        nf_out_o = self.se_nf(nf_out_o) # 现在的 nf_out_o 来自 BRICS
        #改进三
        gate = torch.softmax(self.layer_gate, dim=0)

        final = torch.cat((
            gate[0] * x1_o,
            gate[1] * x2_o,
            # gate[2] * nf_out_o
        ), dim=1)

        a = [int(i) for i in list(idx[0])]
        b = [int(i) for i in list(idx[1])]

        aa = torch.tensor(a, dtype=torch.long)
        bb = torch.tensor(b, dtype=torch.long)
        

        entity1 = final[aa]
        entity2 = final[bb]

        #skip connection
        #entity1_res = self.features1[aa].to('cuda')
        #entity2_res = self.features1[bb].to('cuda')
        entity1_res = self.features1[aa].to('cuda' if torch.cuda.is_available() else 'cpu')
        entity2_res = self.features1[bb].to('cuda' if torch.cuda.is_available() else 'cpu')
        entity1 = torch.cat((entity1, entity1_res), dim=1)
        entity2 = torch.cat((entity2, entity2_res), dim=1)

        concatenate = torch.cat((entity1, entity2), dim=1)

        feature = self.MLP(concatenate, 7)
        log = feature

        #return log, ret_os, nf_loss_o
        return log