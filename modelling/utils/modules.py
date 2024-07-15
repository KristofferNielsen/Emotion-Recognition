import torch
from typing import Union
import numpy as np
from torch import nn, Tensor
from pytorch_lightning import LightningModule 
import torch
import torch.nn.functional as F
import math


class PositionalEmbedding(nn.Module):
    """
        Use the two formulas in the paper to calculate PositionalEmbedding

        input size: [batch_size, seq_length]
        return size: [batch_size, seq_length, dim_vector]

        Args:
            max_len: Maximum length of input sentence
            dim_vector: the dimension of embedding vector for each input word.
    """

    def __init__(self, dim_vector, max_len):
        super().__init__()

        self.dim_vector = dim_vector
        self.max_len = max_len

        pe = torch.zeros(max_len, dim_vector)
        for pos in range(max_len):
            for i in range(0, dim_vector, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2 * i) / dim_vector)))
                pe[pos, i + 1] = math.cos(pos / (10000 ** ((2 * i) / dim_vector)))

        # The size of the pe matrix is [max_len, dim_vector].
        print(f"pe size：{pe.size()}")

        # Register buffer, indicating that this parameter is not updated. Tips: Registering the buffer is equivalent
        # to defining self.pe in__init__, so self.pe can be called in the forward function below, but the parameters
        # will not be updated.
        self.register_buffer('pe', pe)

    def forward(self, x):
        # The input x to the position code is [batch_size, seq_len], where seq_len is the length of the sentence
        batch_size, seq_len = x.size()
        # Returns location information for the number of previous seq_len
        return self.pe[:seq_len, :]

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dropout_prob=0.5):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, stride)
        self.layer_norm = nn.LayerNorm(out_channels)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout_prob)

    def forward(self, x):
        x = self.conv(x)
        x = self.layer_norm(x.transpose(1, 2)).transpose(1, 2)  # Adjust dimensions for LayerNorm
        x = self.activation(x)
        x = self.dropout(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, dim_vector, dim_hidden, dropout=0.1):
        super().__init__()
        self.feedforward = nn.Sequential(
            nn.Linear(dim_vector, dim_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_hidden, dim_vector)
        )

    def forward(self, x):
        out = self.feedforward(x)
        return out

class Seq(nn.Module):
    def __init__(self,d_feat, d_model, nhead, dropout_prob=0.5):
        super(Seq, self).__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=d_feat,
            num_heads=nhead,
            dropout=dropout_prob,
            batch_first=True,
            device='cuda'
        )
        self.norm1 = nn.LayerNorm(d_feat)
        self.dropout1 = nn.Dropout(dropout_prob)

        self.feedforward = FeedForward(d_feat, d_model, dropout_prob)
        self.linear = nn.Linear(d_feat, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_feat)
        self.dropout2 = nn.Dropout(dropout_prob)

    def forward(self, x1,x2):
        out, _ = self.attn(x1, x2, x2)
        out = self.dropout1(out)
        out = self.norm1(x1+out)

        ## 3、 FeedForward
        _x = out
        out = self.feedforward(out)

        ## 4、Add and Norm
        out = self.dropout2(out)
        out = self.norm2(out+_x)
        #out = self.norm2(out+x1)
        #out =self.linear(out)
        return out