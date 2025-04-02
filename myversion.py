"""
AASIST2: Improved Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks
"""

import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import necessary components from original AASIST
from AASIST import GraphAttentionLayer, HtrgGraphAttentionLayer, GraphPool, CONV

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SELU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class Res2Net_block(nn.Module):
    def __init__(self, nb_filts, first=False, scale=4):
        super().__init__()
        self.first = first
        self.scale = scale
        width = nb_filts[1] // scale
        
        if not self.first:
            self.bn1 = nn.BatchNorm2d(num_features=nb_filts[0])
            self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                                  out_channels=nb_filts[1],
                                  kernel_size=(2, 3),
                                  padding=(1, 1),
                                  stride=1)
        else:
            self.conv1 = nn.Conv2d(in_channels=nb_filts[0],
                                  out_channels=nb_filts[1],
                                  kernel_size=(2, 3),
                                  padding=(1, 1),
                                  stride=1)
        
        self.selu = nn.SELU(inplace=True)
        self.bn2 = nn.BatchNorm2d(num_features=nb_filts[1])
        
        # Split channels for hierarchical processing
        self.convs = nn.ModuleList([
            nn.Conv2d(in_channels=width,
                     out_channels=width,
                     kernel_size=(2, 3),
                     padding=(0, 1),
                     stride=1) for _ in range(scale-1)
        ])
        
        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv2d(in_channels=nb_filts[0],
                                           out_channels=nb_filts[1],
                                           padding=(0, 1),
                                           kernel_size=(1, 3),
                                           stride=1)
        else:
            self.downsample = False
            
        self.se = SELayer(nb_filts[1])
        self.mp = nn.MaxPool2d((1, 3))
        
    def forward(self, x):
        identity = x
        
        if not self.first:
            out = self.bn1(x)
            out = self.selu(out)
        else:
            out = x
            
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.selu(out)
        
        # Multi-scale feature extraction
        spx = torch.split(out, out.shape[1] // self.scale, 1)
        sp = spx[0]
        outputs = [sp]
        
        for i in range(self.scale - 1):
            if i == 0:
                sp = self.convs[i](spx[i+1])
            else:
                sp = self.convs[i](sp + spx[i+1])
            outputs.append(sp)
            
        out = torch.cat(outputs, 1)
        
        # Channel attention with SE layer
        out = self.se(out)
        
        if self.downsample:
            identity = self.conv_downsample(identity)
            
        out += identity
        out = self.mp(out)
        
        return out

class AMSoftmaxLoss(nn.Module):
    def __init__(self, s=30.0, m=0.4):
        super(AMSoftmaxLoss, self).__init__()
        self.s = s  # Scale factor
        self.m = m  # Margin
        self.ce = nn.CrossEntropyLoss()
        
    def forward(self, cosine, labels):
        phi = cosine - self.m * torch.ones_like(cosine).scatter_(1, labels.unsqueeze(1), 1)
        return self.ce(self.s * phi, labels)

class Model(nn.Module):
    def __init__(self, d_args):
        super().__init__()
        self.d_args = d_args
        filts = d_args["filts"]
        gat_dims = d_args["gat_dims"]
        pool_ratios = d_args["pool_ratios"]
        temperatures = d_args["temperatures"]
        
        self.conv_time = CONV(out_channels=filts[0],
                            kernel_size=d_args["first_conv"],
                            in_channels=1)
        
        self.first_bn = nn.BatchNorm2d(num_features=1)
        self.drop = nn.Dropout(0.5, inplace=True)
        self.drop_way = nn.Dropout(0.2, inplace=True)
        self.selu = nn.SELU(inplace=True)
        
        # Use Res2Net blocks instead of Residual blocks
        self.encoder = nn.Sequential(
            nn.Sequential(Res2Net_block(nb_filts=filts[1], first=True)),
            nn.Sequential(Res2Net_block(nb_filts=filts[2])),
            nn.Sequential(Res2Net_block(nb_filts=filts[3])),
            nn.Sequential(Res2Net_block(nb_filts=filts[4])),
            nn.Sequential(Res2Net_block(nb_filts=filts[4])),
            nn.Sequential(Res2Net_block(nb_filts=filts[4])))
        
        # The rest of the architecture follows the original AASIST with minor adjustments
        self.pos_S = nn.Parameter(torch.randn(1, 23, filts[-1][-1]))
        self.master1 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        self.master2 = nn.Parameter(torch.randn(1, 1, gat_dims[0]))
        
        # Graph attention layers
        self.GAT_layer_S = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[0])
        self.GAT_layer_T = GraphAttentionLayer(filts[-1][-1], gat_dims[0], temperature=temperatures[1])
        
        self.HtrgGAT_layer_ST11 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST12 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST21 = HtrgGraphAttentionLayer(gat_dims[0], gat_dims[1], temperature=temperatures[2])
        self.HtrgGAT_layer_ST22 = HtrgGraphAttentionLayer(gat_dims[1], gat_dims[1], temperature=temperatures[2])
        
        # Graph pooling
        self.pool_S = GraphPool(pool_ratios[0], gat_dims[0], 0.3)
        self.pool_T = GraphPool(pool_ratios[1], gat_dims[0], 0.3)
        self.pool_hS1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT1 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hS2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        self.pool_hT2 = GraphPool(pool_ratios[2], gat_dims[1], 0.3)
        
        # Output layer modified for AM-Softmax
        self.out_layer = nn.Linear(5 * gat_dims[1], 2, bias=False)
        nn.init.xavier_normal_(self.out_layer.weight)
    
    def forward(self, x, Freq_aug=False):
        x = x.unsqueeze(1)
        x = self.conv_time(x, mask=Freq_aug)
        x = x.unsqueeze(dim=1)
        x = F.max_pool2d(torch.abs(x), (3, 3))
        x = self.first_bn(x)
        x = self.selu(x)
        
        # Feature encoding
        e = self.encoder(x)
        
        # Spectral GAT (GAT-S)
        e_S, _ = torch.max(torch.abs(e), dim=3)  # max along time
        e_S = e_S.transpose(1, 2) + self.pos_S
        gat_S = self.GAT_layer_S(e_S)
        out_S = self.pool_S(gat_S)
        
        # Temporal GAT (GAT-T)
        e_T, _ = torch.max(torch.abs(e), dim=2)  # max along freq
        e_T = e_T.transpose(1, 2)
        gat_T = self.GAT_layer_T(e_T)
        out_T = self.pool_T(gat_T)
        
        # Learnable master nodes
        master1 = self.master1.expand(x.size(0), -1, -1)
        master2 = self.master2.expand(x.size(0), -1, -1)
        
        # Inference path 1
        out_T1, out_S1, master1 = self.HtrgGAT_layer_ST11(out_T, out_S, master=self.master1)
        out_S1 = self.pool_hS1(out_S1)
        out_T1 = self.pool_hT1(out_T1)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST12(out_T1, out_S1, master=master1)
        out_T1 = out_T1 + out_T_aug
        out_S1 = out_S1 + out_S_aug
        master1 = master1 + master_aug
        
        # Inference path 2
        out_T2, out_S2, master2 = self.HtrgGAT_layer_ST21(out_T, out_S, master=self.master2)
        out_S2 = self.pool_hS2(out_S2)
        out_T2 = self.pool_hT2(out_T2)
        out_T_aug, out_S_aug, master_aug = self.HtrgGAT_layer_ST22(out_T2, out_S2, master=master2)
        out_T2 = out_T2 + out_T_aug
        out_S2 = out_S2 + out_S_aug
        master2 = master2 + master_aug
        
        # Apply dropout
        out_T1 = self.drop_way(out_T1)
        out_T2 = self.drop_way(out_T2)
        out_S1 = self.drop_way(out_S1)
        out_S2 = self.drop_way(out_S2)
        master1 = self.drop_way(master1)
        master2 = self.drop_way(master2)
        
        # Competitive aggregation
        out_T = torch.max(out_T1, out_T2)
        out_S = torch.max(out_S1, out_S2)
        master = torch.max(master1, master2)
        
        # Feature aggregation
        T_max, _ = torch.max(torch.abs(out_T), dim=1)
        T_avg = torch.mean(out_T, dim=1)
        S_max, _ = torch.max(torch.abs(out_S), dim=1)
        S_avg = torch.mean(out_S, dim=1)
        
        # Final features
        last_hidden = torch.cat([T_max, T_avg, S_max, S_avg, master.squeeze(1)], dim=1)
        last_hidden = self.drop(last_hidden)
        
        # Normalize features and weights for AM-Softmax
        features = last_hidden
        features_norm = torch.norm(features, p=2, dim=1, keepdim=True)
        features = features.div(features_norm.expand_as(features))
        
        weights = self.out_layer.weight
        weights_norm = torch.norm(weights, p=2, dim=1, keepdim=True)
        weights = weights.div(weights_norm.expand_as(weights))
        
        # Calculate cosine similarities for AM-Softmax
        cosine = F.linear(features, weights)
        
        return last_hidden, cosine
