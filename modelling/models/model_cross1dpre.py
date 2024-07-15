import torch
from typing import Union
import numpy as np
from torch import nn, Tensor
from pytorch_lightning import LightningModule 
import torch
import torch.nn.functional as F
import math
from utils.modules import ConvBlock, Seq


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

class ModelCross1dpre(nn.Module):
    def __init__(self, input_sizes, output_size,nhead, num_classes, dropout_prob,multi,type,a,t,v):
        super(ModelCross1dpre, self).__init__()
        self.a = a
        self.t = t
        self.v = v
        audio_dim   = input_sizes[0]
        text_dim    = input_sizes[1]
        video_dim   = input_sizes[2]
        output_dim1 = num_classes
        output_dim2 = 1
        dropout = dropout_prob
        hidden_dim = 768

        self.mult_at=nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.mult_ta=nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.mult_va=nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.mult_av=nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.mult_tv=nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)
        self.mult_vt=nn.MultiheadAttention(hidden_dim, nhead, dropout=dropout, batch_first=True)


        self.dense = nn.Linear(hidden_dim, hidden_dim)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj1 = nn.Linear(hidden_dim, num_classes)
        self.out_proj2 = nn.Linear(hidden_dim, 1)

        self.fc_att   = nn.Linear(3*hidden_dim, hidden_dim)
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
        at = self.mult_at(hidden[0], hidden[1],hidden[1])[0].squeeze(1)
        ta = self.mult_ta(hidden[1], hidden[0],hidden[0])[0].squeeze(1)
        va = self.mult_va(hidden[2], hidden[0],hidden[0])[0].squeeze(1)
        av = self.mult_av(hidden[0], hidden[2],hidden[2])[0].squeeze(1)
        tv = self.mult_tv(hidden[1], hidden[2],hidden[2])[0].squeeze(1)
        vt = self.mult_vt(hidden[2], hidden[1],hidden[1])[0].squeeze(1)

        a = at*av
        t = ta*tv
        v = va*vt

        fused_feat = self.fc_att(torch.cat((a,t,v),dim=1))
        
        x = self.dense(fused_feat)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x1 = self.out_proj1(x)

        if self.multi:
            x2  = self.fc_out_2(x)
            return x1, x2
        else:
            return x1,a1,t1,v1