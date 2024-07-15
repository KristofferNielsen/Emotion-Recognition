import copy
import torch
import random
import numpy as np
from torch import nn
from torch import einsum
from functools import partial
import torch.nn.functional as F
from typing import Optional, Any
import math
from torch.nn import Parameter

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout

def get_positional_embeddings(seq_length, d_model):
    """
    Generates positional embeddings using sine and cosine functions.

    Args:
    seq_length (int): Length of the sequence.
    d_model (int): Dimension of the embedding.

    Returns:
    torch.Tensor: Positional embeddings matrix of shape (seq_length, d_model) on CUDA.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    position = torch.arange(seq_length, device=device).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * -(torch.log(torch.tensor(10000.0, device=device)) / d_model))
    
    pos_embedding = torch.zeros((seq_length, d_model), device=device)
    pos_embedding[:, 0::2] = torch.sin(position * div_term)
    pos_embedding[:, 1::2] = torch.cos(position * div_term)
    #return as float tensor on CUDA
    return pos_embedding.float()

class MultiheadCrossAttention(nn.Module):
    """Multi-headed attention.

    See "Attention Is All You Need" for more details.
    """

    def __init__(self, embed_dim, num_heads,qdim=None, kdim=None, vdim=None, dropout=0., bias=True,
                 add_bias_kv=False, add_zero_attn=False, self_attention=False,
                 encoder_decoder_attention=False):
        super().__init__()
        self.embed_dim =qdim# embed_dim
        self.qdim = qdim if qdim is not None else embed_dim
        self.kdim = kdim if kdim is not None else embed_dim
        self.vdim = vdim if vdim is not None else embed_dim

    
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads
       

        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"
        self.scaling = self.head_dim ** -0.5

        self.self_attention = self_attention
        self.encoder_decoder_attention = encoder_decoder_attention
  

        self.k_proj_weight = Parameter(torch.Tensor(embed_dim, self.kdim))
        self.v_proj_weight = Parameter(torch.Tensor(embed_dim, self.vdim))
        self.q_proj_weight = Parameter(torch.Tensor(embed_dim, embed_dim))


        if bias:
            self.in_proj_bias = Parameter(torch.Tensor(3 * embed_dim))
        else:
            self.register_parameter('in_proj_bias', None)

        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

        if add_bias_kv:
            self.bias_k = Parameter(torch.Tensor(1, 1, embed_dim))
            self.bias_v = Parameter(torch.Tensor(1, 1, embed_dim))
        else:
            self.bias_k = self.bias_v = None

        self.add_zero_attn = add_zero_attn

        self.reset_parameters()

        self.onnx_trace = False

        self.enable_torch_version = False
        if hasattr(F, "multi_head_attention_forward"):
            self.enable_torch_version = True
        else:
            self.enable_torch_version = False

    def prepare_for_onnx_export_(self):
        self.onnx_trace = True

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj_weight)
        nn.init.xavier_uniform_(self.v_proj_weight)
        nn.init.xavier_uniform_(self.q_proj_weight)

        nn.init.xavier_uniform_(self.out_proj.weight)
        if self.in_proj_bias is not None:
            nn.init.constant_(self.in_proj_bias, 0.)
            nn.init.constant_(self.out_proj.bias, 0.)
        if self.bias_k is not None:
            nn.init.xavier_normal_(self.bias_k)
        if self.bias_v is not None:
            nn.init.xavier_normal_(self.bias_v)

    def forward(self, query, key, value, key_padding_mask=None, incremental_state=None,
                need_weights=True, static_kv=False, attn_mask=None):
        """Input shape: Time x Batch x Channel

        Timesteps can be masked by supplying a T x T mask in the
        `attn_mask` argument. Padding elements can be excluded from
        the key by passing a binary ByteTensor (`key_padding_mask`) with shape:
        batch x src_len, where padding elements are indicated by 1s.
        """

        tgt_len, bsz, embed_dim = query.size()
        assert embed_dim == self.embed_dim
        assert list(query.size()) == [tgt_len, bsz, embed_dim]


        if self.enable_torch_version and not self.onnx_trace and incremental_state is None and not static_kv:
            return F.multi_head_attention_forward(query, key, value,
                                                    self.embed_dim, self.num_heads,
                                                    torch.empty([0]),
                                                    self.in_proj_bias, self.bias_k, self.bias_v,
                                                    self.add_zero_attn, self.dropout,
                                                    self.out_proj.weight, self.out_proj.bias,
                                                    self.training, key_padding_mask, need_weights,
                                                    attn_mask, use_separate_proj_weight=True,
                                                    q_proj_weight=self.q_proj_weight,
                                                    k_proj_weight=self.k_proj_weight,
                                                    v_proj_weight=self.v_proj_weight)

        if self.self_attention:
            # self-attention
            q, k, v = self.in_proj_qkv(query)
        elif self.encoder_decoder_attention:
            # encoder-decoder attention
            q = self.in_proj_q(query)
          
            if key is None:
                assert value is None
                k = v = None
            else:
                k = self.in_proj_k(key)
                v = self.in_proj_v(key)

        else:
            q = self.in_proj_q(query)
            k = self.in_proj_k(key)
            v = self.in_proj_v(value)
        q *= self.scaling

        
        if self.bias_k is not None:
            assert self.bias_v is not None
            k = torch.cat([k, self.bias_k.repeat(1, bsz, 1)])
            v = torch.cat([v, self.bias_v.repeat(1, bsz, 1)])
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, key_padding_mask.new_zeros(key_padding_mask.size(0), 1)], dim=1)
    
        q = q.contiguous().view(tgt_len, bsz * self.num_heads, self.head_dim).transpose(0, 1)
     
       
        if k is not None:
            k = k.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)
            
        if v is not None:
            v = v.contiguous().view(-1, bsz * self.num_heads, self.head_dim).transpose(0, 1)

        src_len = k.size(1)
      
        # This is part of a workaround to get around fork/join parallelism
        # not supporting Optional types.
        if key_padding_mask is not None and key_padding_mask.shape == torch.Size([]):
            key_padding_mask = None

       
        if key_padding_mask is not None:
            assert key_padding_mask.size(0) == bsz
            assert key_padding_mask.size(1) == src_len

        if self.add_zero_attn:
            src_len += 1
            k = torch.cat([k, k.new_zeros((k.size(0), 1) + k.size()[2:])], dim=1)
            v = torch.cat([v, v.new_zeros((v.size(0), 1) + v.size()[2:])], dim=1)
            if attn_mask is not None:
                attn_mask = torch.cat([attn_mask, attn_mask.new_zeros(attn_mask.size(0), 1)], dim=1)
            if key_padding_mask is not None:
                key_padding_mask = torch.cat(
                    [key_padding_mask, torch.zeros(key_padding_mask.size(0), 1).type_as(key_padding_mask)], dim=1)

        attn_weights = torch.bmm(q, k.transpose(1, 2))
        attn_weights = self.apply_sparse_mask(attn_weights, tgt_len, src_len, bsz)


        assert list(attn_weights.size()) == [bsz * self.num_heads, tgt_len, src_len]

     
    
       
        if attn_mask is not None:
            attn_mask = attn_mask.unsqueeze(0)
            if self.onnx_trace:
                attn_mask = attn_mask.repeat(attn_weights.size(0), 1, 1)
            attn_weights += attn_mask

        if key_padding_mask is not None: #Chekthis
            # don't attend to padding symbols
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            if self.onnx_trace:
                attn_weights = torch.where(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    torch.Tensor([float("-Inf")]),
                    attn_weights.float()
                ).type_as(attn_weights)
            else:
                attn_weights = attn_weights.masked_fill(
                    key_padding_mask.unsqueeze(1).unsqueeze(2),
                    float('-inf'),
                )
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        
        attn_weights = F.softmax(
            attn_weights, dim=-1, onnx_trace=self.onnx_trace,
        ).type_as(attn_weights)
        attn_weights = F.dropout(attn_weights, p=self.dropout, training=self.training)

        
        attn = torch.bmm(attn_weights, v)
        
        assert list(attn.size()) == [bsz * self.num_heads, tgt_len, self.head_dim]
        if (self.onnx_trace and attn.size(1) == 1):
            # when ONNX tracing a single decoder step (sequence length == 1)
            # the transpose is a no-op copy before view, thus unnecessary
            attn = attn.contiguous().view(tgt_len, bsz, embed_dim)
        else:
            attn = attn.transpose(0, 1).contiguous().view(tgt_len, bsz, self.embed_dim)
        attn = self.out_proj(attn)

        if need_weights:
            # average attention weights over heads
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len)
            attn_weights = attn_weights.sum(dim=1) / self.num_heads
        else:
            attn_weights = None

        return attn, attn_weights

    def in_proj_qkv(self, query):
        return self._in_proj(query).chunk(3, dim=-1)

    def in_proj_q(self, query):
        bias = self.in_proj_bias
        if bias is not None:
            bias = bias[:self.embed_dim]
        return F.linear(query, self.q_proj_weight, bias)

    def in_proj_k(self, key):
        weight = self.k_proj_weight
        bias = self.in_proj_bias
        if bias is not None:
            bias = bias[self.embed_dim:2 * self.embed_dim]
        return F.linear(key, weight, bias)

    def in_proj_v(self, value):
        weight = self.v_proj_weight
        bias = self.in_proj_bias
        if bias is not None:
            bias = bias[2 * self.embed_dim:]
        return F.linear(value, weight, bias)

    def _in_proj(self, input, start=0, end=None):
        weight = self.in_proj_weight
        bias = self.in_proj_bias
        weight = weight[start:end, :]
        if bias is not None:
            bias = bias[start:end]
        return F.linear(input, weight, bias)

    def apply_sparse_mask(self, attn_weights, tgt_len, src_len, bsz):
        return attn_weights

class TransformerMultiEncoderLayer(nn.Module):
    """
    Implements a Transformer Encoder Layer used in BERT/XLM style pre-trained
    models.
    """

    def __init__(
        self,
        embedding_dim: float = 768,   #need to add the embedding dimentions of other modalities
        qdim :float=768,
        kdim : float = 768,
        vdim : float = 768,
        ffn_embedding_dim: float = 3072,
        num_attention_heads: float = 8,
        dropout: float = 0.1,
        attention_dropout: float = 0.1,
        activation_dropout: float = 0.1,
        add_bias_kv: bool = False,
        add_zero_attn: bool = False,
        self_attention:bool=False,
        encoder_decoder_attention:bool=False
    ) -> None:

        super().__init__()
        # Initialize parameters
        self.embedding_dim = embedding_dim
        self.query_dim = qdim
        self.key_dim = kdim
        self.value_dim = vdim

        self.self_attention=self_attention
        self.encoder_decorder_attention=encoder_decoder_attention

        self.dropout = dropout
        self.activation_dropout = activation_dropout

        # Initialize blocks
        self.activation_fn = F.relu

        self.self_attn = MultiheadCrossAttention( 
            self.embedding_dim,
            num_attention_heads,
            dropout=attention_dropout,
            add_bias_kv=add_bias_kv,
            add_zero_attn=add_zero_attn,
            self_attention=self.self_attention,
            kdim=self.key_dim,
            vdim=self.value_dim,
            qdim=self.query_dim,
            encoder_decoder_attention=self.encoder_decorder_attention
        )

        # layer norm associated with the self attention layer
        self.self_attn_layer_norm = nn.LayerNorm(self.embedding_dim)
        self.fc1 = nn.Linear(self.embedding_dim, 3072)
        self.fc2 = nn.Linear(3072, self.embedding_dim)

        # layer norm associated with the position wise feed-forward NN
        self.final_layer_norm = nn.LayerNorm(self.embedding_dim)

    def forward(   #If we are modifying this we need three modalities
        self,
        # x: torch.Tensor,  #This is dictionary consist of text,audio,video
        x_q: torch.Tensor,   #The modality to calculate the query
        x_k: torch.Tensor,   #The modality to calculate the key
        x_v: torch.Tensor, #The modality to calculate the value (for future)
        self_attn_mask: torch.Tensor = None,
        self_attn_padding_mask: torch.Tensor = None,    
    ):
        """
        LayerNorm is applied either before or after the self-attention/ffn
        modules similar to the original Transformer imlementation.
        """
        # residual = x
        
        residual = x_q

        x, attn = self.self_attn(
            query=x_q,
            key=x_k,
            value=x_v,
            key_padding_mask=self_attn_padding_mask,
            need_weights=False,
            attn_mask=self_attn_mask,
        )

        

        #x=x_cr    
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.self_attn_layer_norm(x)

        residual = x
        x = self.activation_fn(self.fc1(x))
        x = F.dropout(x, p=self.activation_dropout, training=self.training)
        x = self.fc2(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = residual + x
        x = self.final_layer_norm(x)
        return x, attn
        

class Had2d(nn.Module):
    def __init__(self, input_sizes, output_size,nhead, num_classes, dropout_prob,multi,type,a,t,v):
        super(Had2d, self).__init__()

        ffn_embedding_dim: int = 3072
        attention_dropout: float = 0.1
        activation_dropout: float = 0.1
        self.dropout = dropout_prob

        self.audio_width = audio_width = input_sizes[0]
        self.text_width = text_width = input_sizes[1]
        self.vision_width = vision_width = input_sizes[2]
        #self.embed_dim = embed_dim = output_size
        
        self.AudioLayer= nn.TransformerEncoderLayer(d_model=audio_width, nhead=nhead, dropout=dropout_prob,batch_first=True)
        self.audio_encoder= nn.TransformerEncoder(self.AudioLayer, num_layers=1)

        self.TextLayer= nn.TransformerEncoderLayer(d_model=text_width, nhead=nhead, dropout=dropout_prob,batch_first=True)  
        self.text_encoder= nn.TransformerEncoder(self.TextLayer, num_layers=1)

        self.VideoLayer= nn.TransformerEncoderLayer(d_model=vision_width, nhead=nhead, dropout=dropout_prob,batch_first=True)
        self.visual_encoder= nn.TransformerEncoder(self.VideoLayer, num_layers=1)


        self.vt_layers = nn.ModuleList([TransformerMultiEncoderLayer(
                            embedding_dim=self.vision_width,
                            qdim=self.vision_width,
                            kdim=self.text_width,
                            vdim=self.text_width,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=nhead,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            add_bias_kv=False,
                            add_zero_attn=False) for _ in range(1)])
        
        self.tv_layers = nn.ModuleList([TransformerMultiEncoderLayer(
                            embedding_dim=self.text_width,
                            qdim=self.text_width,
                            kdim=self.vision_width,
                            vdim=self.vision_width,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=nhead,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            add_bias_kv=False,
                            add_zero_attn=False) for _ in range(1)])
        
        self.at_layers = nn.ModuleList([TransformerMultiEncoderLayer(
                            embedding_dim=self.audio_width,
                            qdim=self.audio_width,
                            kdim=self.text_width,
                            vdim=self.text_width,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=nhead,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            add_bias_kv=False,
                            add_zero_attn=False) for _ in range(1)])
        
        self.ta_layers = nn.ModuleList([TransformerMultiEncoderLayer(
                            embedding_dim=self.text_width,
                            qdim=self.text_width,
                            kdim=self.audio_width,
                            vdim=self.audio_width,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=nhead,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            add_bias_kv=False,
                            add_zero_attn=False) for _ in range(1)])
        
        self.av_layers = nn.ModuleList([TransformerMultiEncoderLayer(
                            embedding_dim=self.audio_width,
                            qdim=self.audio_width,
                            kdim=self.vision_width,
                            vdim=self.vision_width,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=nhead,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            add_bias_kv=False,
                            add_zero_attn=False) for _ in range(1)])
        
        self.va_layers = nn.ModuleList([TransformerMultiEncoderLayer(
                            embedding_dim=self.vision_width,
                            qdim=self.vision_width,
                            kdim=self.audio_width,
                            vdim=self.audio_width,
                            self_attention=False,
                            encoder_decoder_attention=True,
                            ffn_embedding_dim=ffn_embedding_dim,
                            num_attention_heads=nhead,
                            dropout=self.dropout,
                            attention_dropout=attention_dropout,
                            activation_dropout=activation_dropout,
                            add_bias_kv=False,
                            add_zero_attn=False) for _ in range(1)])
        
        self.dense = nn.Linear(vision_width+audio_width+text_width, 768)
        self.activation_fn = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_prob)
        self.out_proj = nn.Linear(768, num_classes)
        self.out_proj1 = nn.Linear(768,1)


    def forward(self, audio_input, audio_input_mask, text_input, text_input_mask,video_input, video_input_mask):
        #positional encoding
        #audio_input  = audio_input + get_positional_embeddings(audio_input.size(1), audio_input.size(2))
        #text_input = text_input + get_positional_embeddings(text_input.size(1), text_input.size(2))
        #video_input = video_input + get_positional_embeddings(video_input.size(1), video_input.size(2))

        #layer norm
        #audio_input = F.layer_norm(audio_input, audio_input.size()[1:])
        #text_input = F.layer_norm(text_input, text_input.size()[1:])
        #video_input = F.layer_norm(video_input, video_input.size()[1:])

        #self attention
        image_embeds = self.visual_encoder(video_input, src_key_padding_mask=video_input_mask).permute(1, 0, 2)
        audio_embeds = self.audio_encoder(audio_input, src_key_padding_mask=audio_input_mask).permute(1, 0, 2)
        text_embeds = self.text_encoder(text_input, src_key_padding_mask=text_input_mask).permute(1, 0, 2)

        image_embeds = self.dropout1(image_embeds)
        audio_embeds = self.dropout1(audio_embeds)
        text_embeds = self.dropout1(text_embeds)

        #image_q = torch.mean(image_embeds, dim=0).unsqueeze(0)
        #audio_q = torch.mean(audio_embeds, dim=0).unsqueeze(0)
        #text_q = torch.mean(text_embeds, dim=0).unsqueeze(0)

        #cross attention
        for i in range(1):
            x_vt, _ = self.vt_layers[i](image_embeds, text_embeds, text_embeds, self_attn_padding_mask=text_input_mask)
            x_tv, _ = self.tv_layers[i](text_embeds, image_embeds, image_embeds, self_attn_padding_mask=video_input_mask)
            x_at, _ = self.at_layers[i](audio_embeds, text_embeds, text_embeds, self_attn_padding_mask=text_input_mask)
            x_ta, _ = self.ta_layers[i](text_embeds, audio_embeds, audio_embeds, self_attn_padding_mask=audio_input_mask)
            x_av, _ = self.av_layers[i](audio_embeds, image_embeds, image_embeds, self_attn_padding_mask=video_input_mask)
            x_va, _ = self.va_layers[i](image_embeds, audio_embeds, audio_embeds, self_attn_padding_mask=audio_input_mask)

        x_vt = torch.mean(x_vt,dim=0)
        x_tv = torch.mean(x_tv,dim=0)
        x_at = torch.mean(x_at,dim=0)
        x_ta = torch.mean(x_ta,dim=0)
        x_av = torch.mean(x_av,dim=0)
        x_va = torch.mean(x_va,dim=0)

        v = torch.mul(x_vt,x_va)
        a = torch.mul(x_at,x_av)
        t = torch.mul(x_tv,x_ta)

        z = torch.cat([v, a, t], dim=1)

        z = self.dense(z)
        z = self.activation_fn(z)
        z = self.dropout1(z)
        z1 = self.out_proj(z)
        #z2 = self.out_proj1(z)

        return z1.squeeze(0)#, z2.squeeze(0)