import torch
from typing import Union
import numpy as np
from torch import nn, Tensor
from pytorch_lightning import LightningModule 
import torch
import torch.nn.functional as F
import math
from utils.modules import ConvBlock, Seq

class Mult(nn.Module):
    def __init__(self, input_sizes, output_size, num_classes,nhead, dropout_prob,multi,a,t,v):
        super(Mult, self).__init__()
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

        self.audio_conv = ConvBlock(in_channels=audio_dim, out_channels=hidden_dim, dropout_prob=dropout)
        self.text_conv = ConvBlock(in_channels=text_dim, out_channels=hidden_dim, dropout_prob=dropout)
        self.visual_conv = ConvBlock(in_channels=video_dim, out_channels=hidden_dim, dropout_prob=dropout)

        self.cls_audio = nn.Embedding(1, hidden_dim)
        self.cls_text = nn.Embedding(1, hidden_dim)
        self.cls_video  = nn.Embedding(1, hidden_dim)
        
        self.AudioLayer= nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.AudioTransformer= nn.TransformerEncoder(self.AudioLayer, num_layers=2)

        self.TextLayer= nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)  
        self.TextTransformer= nn.TransformerEncoder(self.TextLayer, num_layers=2)

        self.VideoLayer= nn.TransformerEncoderLayer(d_model=hidden_dim, nhead=nhead, dropout=dropout,batch_first=True)
        self.VideoTransformer= nn.TransformerEncoder(self.VideoLayer, num_layers=2)

        self.pred = nn.Linear(hidden_dim, output_dim1)
        self.pred2 = nn.Linear(hidden_dim, output_dim2)
    
    def forward(self, mods):
        batch_size = mods[0].shape[0]
        audio_cls = self.cls_audio(torch.zeros(batch_size,dtype=torch.long,device='cuda'))
        text_cls = self.cls_text(torch.zeros(batch_size,dtype=torch.long,device='cuda'))
        video_cls = self.cls_video(torch.zeros(batch_size,dtype=torch.long,device='cuda'))

        #audio = self.audio_conv(mods[0].permute(0,2,1)).permute(0,2,1)
        #text = self.text_conv(mods[0].permute(0,2,1)).permute(0,2,1)
        image = self.visual_conv(mods[0].permute(0,2,1)).permute(0,2,1)

        #audio = torch.cat((audio_cls.unsqueeze(1), audio), dim=1)
        #text = torch.cat((text_cls.unsqueeze(1), text), dim=1)
        video = torch.cat((video_cls.unsqueeze(1), image), dim=1)

        if self.a:
            audio = self.AudioTransformer(audio)
            preds_audio = self.pred(audio[:,0,:])
        if self.t:
            text = self.TextTransformer(text)
            preds_text = self.pred(text[:,0,:])
        if self.v:
            video = self.VideoTransformer(video)
            preds_video = self.pred(video[:,0,:])

        return preds_video
       