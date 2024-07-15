import torch
from torch import nn
import torch
import torch.nn.functional as F
import math

class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = self.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )
        self.register_buffer('_float_tensor', torch.FloatTensor(1))

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings."""
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float32).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, incremental_state=None, timestep=None):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size()
        max_pos = self.padding_idx + 1 + seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = self.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(self._float_tensor.device)

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return self.weights[self.padding_idx + pos, :].expand(bsz, 1, -1)

        positions = self.make_positions(input, self.padding_idx)
        return self.weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1)

    @staticmethod
    def make_positions(tensor, padding_idx):
        """Replace non-padding symbols with their position numbers.
        Position numbers begin at padding_idx+1. Padding symbols are ignored."""
        mask = tensor.ne(padding_idx).int()
        return (torch.cumsum(mask, dim=1) * mask).long() + padding_idx

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number


class ModelCLS(nn.Module):
    def __init__(self, input_dims, d_model, nhead, num_classes, dropout=0.4):
        super(ModelCLS, self).__init__()
        self.d_model = d_model
        self.nhead = nhead
        
        # Transformer encoder layer
        encoder_layer_audio = nn.TransformerEncoderLayer(d_model=input_dims[0], nhead=nhead, dropout=dropout,batch_first=True)
        encoder_layer_text = nn.TransformerEncoderLayer(d_model=input_dims[1], nhead=nhead, dropout=dropout,batch_first=True)
        encoder_layer_video = nn.TransformerEncoderLayer(d_model=input_dims[2], nhead=nhead, dropout=dropout,batch_first=True)
        self.transformer_encoder_audio = nn.TransformerEncoder(encoder_layer_audio, num_layers=2)
        self.transformer_encoder_text = nn.TransformerEncoder(encoder_layer_text, num_layers=2)
        self.transformer_encoder_video = nn.TransformerEncoder(encoder_layer_video, num_layers=2)

        #multihead attention
        self.mult_at=nn.MultiheadAttention(input_dims[0], nhead, dropout=dropout, batch_first=True,kdim=input_dims[1],vdim=input_dims[1])
        self.mult_ta=nn.MultiheadAttention(input_dims[1], nhead, dropout=dropout, batch_first=True,kdim=input_dims[0],vdim=input_dims[0])
        self.mult_va=nn.MultiheadAttention(input_dims[2], nhead, dropout=dropout, batch_first=True,kdim=input_dims[0],vdim=input_dims[0])
        self.mult_av=nn.MultiheadAttention(input_dims[0], nhead, dropout=dropout, batch_first=True,kdim=input_dims[2],vdim=input_dims[2])
        self.mult_tv=nn.MultiheadAttention(input_dims[1], nhead, dropout=dropout, batch_first=True,kdim=input_dims[2],vdim=input_dims[2])
        self.mult_vt=nn.MultiheadAttention(input_dims[2], nhead, dropout=dropout, batch_first=True,kdim=input_dims[1],vdim=input_dims[1])
        
        # CLS token            
        self.cls_audio = nn.Embedding(1, input_dims[0])
        self.cls_text = nn.Embedding(1, input_dims[1])
        self.cls_video  = nn.Embedding(1, input_dims[2])
        
        # Positional encoding
        self.positional_encoding_audio = SinusoidalPositionalEmbedding(input_dims[0], 0)
        self.positional_encoding_text = SinusoidalPositionalEmbedding(input_dims[1], 0)
        self.positional_encoding_video = SinusoidalPositionalEmbedding(input_dims[2], 0)

        # Fully connected layer
        self.dense = nn.Linear(input_dims[0]+input_dims[1]+input_dims[2], d_model)
        self.activation_fn = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj1 = nn.Linear(d_model, num_classes)
        self.out_proj2 = nn.Linear(d_model, 1)


    def forward(self, mods):
        batch_size = mods[0].shape[0]
        
        # Expand and concatenate CLS token to embeddings
        audio_cls =  torch.tensor(mods[0].size(0),1,dtype=torch.long,device='cuda').fill_(0)
        text_cls =  torch.tensor(mods[1].size(0),1,dtype=torch.long,device='cuda').fill_(0)
        video_cls =  torch.tensor(mods[2].size(0),1,dtype=torch.long,device='cuda').fill_(0)

        audio = torch.cat((self.cls_audio(audio_cls), mods[0]), dim=1)
        text = torch.cat((self.cls_text(text_cls), mods[1]), dim=1)
        video = torch.cat((self.cls_video(video_cls), mods[2]), dim=1)

        # Add positional encoding to embeddings
        audio_pos = self.positional_encoding_audio(audio[:,:,0])
        text_pos = self.positional_encoding_text(text[:,:,0])
        video_pos = self.positional_encoding_video(video[:,:,0])

        audio = audio + audio_pos
        text = text + text_pos
        video = video + video_pos
        
        # Apply transformer encoder
        audio = self.transformer_encoder_audio(audio)
        text = self.transformer_encoder_text(text)
        video = self.transformer_encoder_video(video)

        #multihead attention with cls as query
        at, _ = self.mult_at(audio[:,0,:].unsqueeze(1), text, text)
        ta, _ = self.mult_ta(text[:,0,:].unsqueeze(1), audio, audio)
        va, _ = self.mult_va(video[:,0,:].unsqueeze(1), audio, audio)
        av, _ = self.mult_av(audio[:,0,:].unsqueeze(1), video, video)
        tv, _ = self.mult_tv(text[:,0,:].unsqueeze(1), video, video)
        vt, _ = self.mult_vt(video[:,0,:].unsqueeze(1), text, text)

        #hadamard product
        a = at*av
        t = ta*tv
        v = va*vt

        #concatenate
        x = torch.cat((a.squeeze(1),t.squeeze(1),v.squeeze(1)),dim=1)

        # Fully connected layer
        x = self.dense(x)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x1 = self.out_proj1(x)
        x2 = self.out_proj2(x)
        return x1, x2