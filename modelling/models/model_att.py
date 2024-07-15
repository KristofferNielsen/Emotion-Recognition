import torch
from typing import Union
import numpy as np
from torch import nn, Tensor
from pytorch_lightning import LightningModule 
import torch
import torch.nn.functional as F
import math

class MLPEncoder(nn.Module):
    '''
    The subnetwork that is used in TFN for video and audio in the pre-fusion stage
    '''

    def __init__(self, in_size, hidden_size, dropout):
        '''
        Args:
            in_size: input dimension
            hidden_size: hidden layer dimension
            dropout: dropout probability
        Output:
            (return value in forward) a tensor of shape (batch_size, hidden_size)
        '''
        super(MLPEncoder, self).__init__()
        # self.norm = nn.BatchNorm1d(in_size)
        self.drop = nn.Dropout(p=dropout)
        self.linear_1 = nn.Linear(in_size, hidden_size)
        self.linear_2 = nn.Linear(hidden_size, hidden_size)
        self.linear_3 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, in_size)
        '''
        # normed = self.norm(x)
        dropped = self.drop(x)
        y_1 = F.relu(self.linear_1(dropped))
        y_2 = F.relu(self.linear_2(y_1))
        y_3 = F.relu(self.linear_3(y_2))

        return y_3

class Attention(nn.Module):
    def __init__(self, input_sizes, output_size, num_classes, dropout_prob,multi,type,a,t,v):
        super(Attention, self).__init__()
        self.a = a
        self.t = t
        self.v = v
        audio_dim   = input_sizes[0]
        text_dim    = input_sizes[1]
        video_dim   = input_sizes[2]
        output_dim1 = num_classes
        output_dim2 = 1
        dropout = dropout_prob
        hidden_dim = output_size
        
        self.audio_encoder = MLPEncoder(audio_dim, hidden_dim, dropout)
        self.text_encoder  = MLPEncoder(text_dim,  hidden_dim, dropout)
        self.video_encoder = MLPEncoder(video_dim, hidden_dim, dropout)

        self.attention_mlp = MLPEncoder(hidden_dim * (a+t+v), hidden_dim, dropout)

        self.fc_att   = nn.Linear(hidden_dim, (a+t+v))
        self.fc_out_1 = nn.Linear(hidden_dim, output_dim1)
        self.fc_out_2 = nn.Linear(hidden_dim, output_dim2)
        self.multi = multi
    
    def forward(self, mods):
        '''
            support feat_type: utt | frm-align | frm-unalign
        '''
        hidden = []
        count = 0
        if self.a:
            audio_hidden = self.audio_encoder(mods[count]) # [32, 128]
            hidden.append(audio_hidden)
            count +=1
        if self.t:
            text_hidden  = self.text_encoder(mods[count])  # [32, 128]
            hidden.append(text_hidden)
            count+=1
        if self.v:
            video_hidden = self.video_encoder(mods[count]) # [32, 128]
            hidden.append(video_hidden)
            count+=1
        if count==1:
            emos_out  = self.fc_out_1(hidden[0])
            if self.multi:
                vals_out  = self.fc_out_2(hidden[0])
                return emos_out, vals_out
            else:
                return emos_out
        
        else:
            multi_hidden1 = torch.cat(hidden, dim=1) # [32, 384]
            attention = self.attention_mlp(multi_hidden1)
            attention = self.fc_att(attention)
            attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

            multi_hidden2 = torch.stack(hidden, dim=2) # [32, 128, 3]
            fused_feat = torch.matmul(multi_hidden2, attention)  # [32, 128, 3] * [32, 3, 1] = [32, 128, 1]

            features  = fused_feat.squeeze(axis=2) # [32, 128] => 解决batch=1报错的问题
            emos_out  = self.fc_out_1(features)
            if self.multi:
                vals_out  = self.fc_out_2(features)
                return emos_out, vals_out
            else:
                return emos_out