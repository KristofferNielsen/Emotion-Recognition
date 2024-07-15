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

class LSTMEncoder(nn.Module):
    '''
    The LSTM-based subnetwork that is used in TFN for text
    '''

    def __init__(self, in_size, hidden_size, dropout, num_layers=1, bidirectional=False):

        super(LSTMEncoder, self).__init__()

        if num_layers == 1:
            rnn_dropout = 0.0
        else:
            rnn_dropout = dropout

        self.rnn = nn.LSTM(in_size, hidden_size, num_layers=num_layers, dropout=rnn_dropout, bidirectional=bidirectional, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_1 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        '''
        Args:
            x: tensor of shape (batch_size, sequence_len, in_size)
            因为用的是 final_states ，所以特征的 padding 是放在前面的
        '''
        _, final_states = self.rnn(x)
        h = self.dropout(final_states[0].squeeze(0))
        y_1 = self.linear_1(h)
        return y_1

class Attentionpre(nn.Module):
    def __init__(self, input_sizes, output_size, num_classes, dropout_prob,multi,type,a,t,v):
        super(Attentionpre, self).__init__()
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
        hidden.append(mods[0])
        hidden.append(mods[1])
        hidden.append(mods[2])
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