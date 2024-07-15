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

class FeedForward(nn.Module):
    def __init__(self, dim, mult=4, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult),
            nn.GELU(),
            nn.Linear(dim * mult, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class MultimodalModel(nn.Module):
    def __init__(self, input_dims, d_model, nhead, num_classes, dropout):
        super(MultimodalModel, self).__init__()
        self.audio_encoder = nn.Linear(input_dims[0], d_model)
        self.text_encoder= nn.Linear(input_dims[1], d_model)
        self.video_encoder = nn.Linear(input_dims[2], d_model)

        self.mult_as=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mult_sa=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mult_ts=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mult_st=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mult_vs=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.mult_sv=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.self_as=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_sa=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_ts=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_st=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_vs=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.self_sv=nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)

        self.ff_as = FeedForward(d_model,dropout=dropout)
        self.ff_sa = FeedForward(d_model,dropout=dropout)
        self.ff_ts = FeedForward(d_model,dropout=dropout)
        self.ff_st = FeedForward(d_model,dropout=dropout)
        self.ff_vs = FeedForward(d_model,dropout=dropout)
        self.ff_sv = FeedForward(d_model,dropout=dropout)


        self.layer_norm = nn.LayerNorm(d_model)
        
        self.attention_mlp1 = MLPEncoder(d_model*3, d_model, dropout)
        self.fc_att   = nn.Linear(d_model, 3)
        self.fc_out_1 = nn.Linear(d_model, num_classes)
        self.fc_out_2 = nn.Linear(d_model, 1)

        self.attention_mlp2 = MLPEncoder(d_model*3, d_model, dropout)
        self.fc_att2   = nn.Linear(d_model, 3)
        self.fc_out_3 = nn.Linear(d_model, num_classes)
        self.fc_out_4 = nn.Linear(d_model, 1)



    def forward(self, mods):
        audio1d = self.audio_encoder(torch.mean(mods[0], dim=1))
        text1d = self.text_encoder(torch.mean(mods[1], dim=1))
        video1d = self.video_encoder(torch.mean(mods[2], dim=1))
        stack = torch.stack([audio1d, text1d, video1d], dim=1)
        stack = self.layer_norm(stack)
            
        audio2d = self.audio_encoder(mods[0])
        text2d = self.text_encoder(mods[1])
        video2d = self.video_encoder(mods[2])

        audio2d = self.layer_norm(audio2d)
        text2d = self.layer_norm(text2d)
        video2d = self.layer_norm(video2d)

        #audio,_ = self.mult_as(audio2d, stack, stack)
        #text,_ = self.mult_ts(text2d, stack, stack)
        #video,_ = self.mult_vs(video2d, stack, stack)

        #audio = self.layer_norm(audio+audio2d)
        #text = self.layer_norm(text+text2d)
        #video = self.layer_norm(video+video2d)

        #audio = self.layer_norm(self.self_as(audio, audio, audio)[0]+audio)
        #text = self.layer_norm(self.self_ts(text, text, text)[0]+text)
        #video = self.layer_norm(self.self_vs(video, video, video)[0]+video)

        #audio = self.layer_norm(self.ff_as(audio)+audio)
        #text = self.layer_norm(self.ff_ts(text)+text)
        #video = self.layer_norm(self.ff_vs(video)+video)

        #avg pool and concat
        #audio = torch.mean(audio, dim=1)
        #text = torch.mean(text, dim=1)
        #video = torch.mean(video, dim=1)

        #multi_hidden1 = torch.cat([audio, text, video], dim=1)
        #attention = self.attention_mlp1(multi_hidden1)
        #attention = self.fc_att(attention)
        #attention = torch.unsqueeze(attention, 2) # [32, 3, 1]

        #multi_hidden2 = torch.stack([audio, text, video], dim=2) # [32, 128, 3]
        #fused_feat = torch.matmul(multi_hidden2, attention)  # [32, 128, 3] * [32, 3, 1] = [32, 128, 1]
        #features  = fused_feat.squeeze(axis=2)
        #out1 = self.fc_out_1(features)
        #out2 = self.fc_out_2(features)
    

        stack1,_ = self.mult_sa(stack, audio2d, audio2d)
        stack2,_ = self.mult_st(stack, text2d, text2d)
        stack3,_ = self.mult_sv(stack, video2d, video2d)

        stack1 = self.layer_norm(stack1+stack)
        stack2 = self.layer_norm(stack2+stack)
        stack3 = self.layer_norm(stack3+stack)

        stack1 = self.layer_norm(self.self_sa(stack1, stack1, stack1)[0]+stack1)
        stack2 = self.layer_norm(self.self_st(stack2, stack2, stack2)[0]+stack2)
        stack3 = self.layer_norm(self.self_sv(stack3, stack3, stack3)[0]+stack3)

        stack1 = self.layer_norm(self.ff_sa(stack1)+stack1)
        stack2 = self.layer_norm(self.ff_st(stack2)+stack2)
        stack3 = self.layer_norm(self.ff_sv(stack3)+stack3)

        #avg pool and concat
        audio2d = torch.mean(stack1, dim=1)
        text2d = torch.mean(stack2, dim=1)
        video2d = torch.mean(stack3, dim=1)

        multi_hidden1 = torch.cat([audio2d, text2d, video2d], dim=1)
        attention = self.attention_mlp2(multi_hidden1)
        attention = self.fc_att2(attention)
        attention = torch.unsqueeze(attention, 2)

        multi_hidden2 = torch.stack([audio2d, text2d, video2d], dim=2)
        fused_feat = torch.matmul(multi_hidden2, attention)
        features  = fused_feat.squeeze(axis=2)
        out3 = self.fc_out_3(features)
        out4 = self.fc_out_4(features)

        return out3, out4#, out1, out2