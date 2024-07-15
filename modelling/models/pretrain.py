import copy
import torch
import random
import numpy as np
from torch import nn
from torch import einsum
from functools import partial
import torch.nn.functional as F
from typing import Optional, Any

from torch import Tensor
import torch.nn.functional as F
from torch.nn.modules import Module
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from transformers import BertModel, BertConfig
from torch.nn.modules.container import ModuleList
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import MultiheadAttention

class Pretrain(nn.Module):
    def __init__(self, input_sizes, output_size,nhead, num_classes, dropout_prob,multi,type,a,t,v):
        super(Pretrain, self).__init__()
        self.vision_width = vision_width = 768#input_sizes[2]
        self.audio_width = audio_width = 768#input_sizes[0]
        self.text_width = text_width = 768#input_sizes[1]
        self.embed_dim = embed_dim = 768
        
        ############################################# set unimodal TextEncoder #############################################
        # contrastive projection 
        self.vision_proj1 = nn.Linear(input_sizes[2], 768)
        self.text_proj1 = nn.Linear(input_sizes[1], 768)
        self.audio_proj1 = nn.Linear(input_sizes[0], 768)

        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)
        self.audio_proj = nn.Linear(audio_width, embed_dim)

        self.AudioLayer= nn.TransformerEncoderLayer(d_model=audio_width, nhead=nhead, dropout=dropout_prob,batch_first=True)
        self.audio_encoder= nn.TransformerEncoder(self.AudioLayer, num_layers=2)

        self.TextLayer= nn.TransformerEncoderLayer(d_model=text_width, nhead=nhead, dropout=dropout_prob,batch_first=True)  
        self.text_encoder= nn.TransformerEncoder(self.TextLayer, num_layers=2)

        self.VideoLayer= nn.TransformerEncoderLayer(d_model=vision_width, nhead=nhead, dropout=dropout_prob,batch_first=True)
        self.visual_encoder= nn.TransformerEncoder(self.VideoLayer, num_layers=2)

        self.temp = nn.Parameter(torch.ones([]) * 0.07)   
        self.queue_size = 65536
        self.momentum = 0.995
        self.pred_head = nn.Linear(embed_dim*3, 2)

        self.AudioLayer_m= nn.TransformerEncoderLayer(d_model=audio_width, nhead=nhead, dropout=dropout_prob,batch_first=True)
        self.audio_encoder_m= nn.TransformerEncoder(self.AudioLayer, num_layers=2)

        self.TextLayer_m= nn.TransformerEncoderLayer(d_model=text_width, nhead=nhead, dropout=dropout_prob,batch_first=True)  
        self.text_encoder_m= nn.TransformerEncoder(self.TextLayer, num_layers=2)

        self.VideoLayer= nn.TransformerEncoderLayer(d_model=vision_width, nhead=nhead, dropout=dropout_prob,batch_first=True)
        self.visual_encoder_m= nn.TransformerEncoder(self.VideoLayer, num_layers=2)

        # create momentum models
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.audio_proj_m = nn.Linear(audio_width, embed_dim)     
        self.text_proj_m = nn.Linear(text_width, embed_dim)   
        
        self.model_pairs = [
                [self.visual_encoder,self.visual_encoder_m],
                [self.vision_proj,self.vision_proj_m],
                [self.audio_encoder,self.audio_encoder_m],
                [self.audio_proj,self.audio_proj_m],
                [self.text_encoder,self.text_encoder_m],
                [self.text_proj,self.text_proj_m],]
        
        
        self.copy_params()
        # create the queue

        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("audio_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))  
                             
        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.audio_queue = nn.functional.normalize(self.audio_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)

    def forward(self, audio_input, audio_input_mask, text_input, text_input_mask,video_input, video_input_mask,  alpha=0.4, is_train=True):
        with torch.no_grad():
            self.temp.clamp_(0.001,0.5)    

        video_input=self.vision_proj1(video_input)
        image_embeds = self.visual_encoder(video_input, src_key_padding_mask=video_input_mask) # image_embeds.shape torch.Size([batchsize, 5408, vision_width])
        image_mask_expanded = (video_input_mask).float().unsqueeze(-1).expand(image_embeds.size())
        image_sum_embeddings = torch.sum(image_embeds * image_mask_expanded, axis=1)
        image_sum_mask = image_mask_expanded.sum(axis=1)
        image_sum_mask = torch.clamp(image_sum_mask, min=1e-9)
        mean_image_feat = image_sum_embeddings / image_sum_mask # torch.Size([1, vision_width]
        image_feat = F.normalize(self.vision_proj(mean_image_feat),dim=-1)

        audio_input=self.audio_proj1(audio_input)
        audio_embeds = self.audio_encoder(audio_input, src_key_padding_mask=audio_input_mask)
        audio_mask_expanded = (audio_input_mask).float().unsqueeze(-1).expand(audio_embeds.size())
        audio_sum_embeddings = torch.sum(audio_embeds * audio_mask_expanded, axis=1)
        audio_sum_mask = audio_mask_expanded.sum(axis=1)
        audio_sum_mask = torch.clamp(audio_sum_mask, min=1e-9)
        mean_audio_feat = audio_sum_embeddings / audio_sum_mask # torch.Size([1, audio_width]
        audio_feat = F.normalize(self.audio_proj(mean_audio_feat),dim=-1)
       
        text_input=self.text_proj1(text_input)
        text_embeds = self.text_encoder(text_input, src_key_padding_mask=text_input_mask)
        text_mask_expanded = (text_input_mask).float().unsqueeze(-1).expand(text_embeds.size())
        text_sum_embeddings = torch.sum(text_embeds * text_mask_expanded, axis=1)
        text_sum_mask = text_mask_expanded.sum(axis=1)
        text_sum_mask = torch.clamp(text_sum_mask, min=1e-9)
        mean_text_feat = text_sum_embeddings / text_sum_mask # torch.Size([1, text_width]
        text_feat = F.normalize(self.text_proj(mean_text_feat),dim=-1)
        if is_train:
            # get momentum features
            with torch.no_grad():
                self._momentum_update()

                image_embeds_m = self.visual_encoder_m(video_input, src_key_padding_mask=video_input_mask)
                audio_embeds_m = self.audio_encoder_m(audio_input, src_key_padding_mask=audio_input_mask)      
                text_embeds_m = self.text_encoder_m(text_input, src_key_padding_mask=text_input_mask)

                # text_feat = F.normalize(self.text_proj(text_embeds[:,0,:]),dim=-1)  
                
                audio_mask_expanded_m = (audio_input_mask).float().unsqueeze(-1).expand(audio_embeds_m.size())
                text_mask_expanded_m = (text_input_mask).float().unsqueeze(-1).expand(text_embeds_m.size())
                image_mask_expanded_m = (video_input_mask).float().unsqueeze(-1).expand(image_embeds_m.size())
                text_sum_embeddings_m = torch.sum(text_embeds_m * text_mask_expanded_m, axis=1)
                audio_sum_embeddings_m = torch.sum(audio_embeds_m * audio_mask_expanded_m, axis=1)
                image_sum_embeddings_m = torch.sum(image_embeds_m * image_mask_expanded_m, axis=1)
                text_sum_mask_m = text_mask_expanded_m.sum(axis=1)
                text_sum_mask_m = torch.clamp(text_sum_mask_m, min=1e-9)
                audio_sum_mask_m = audio_mask_expanded_m.sum(axis=1)
                audio_sum_mask_m = torch.clamp(audio_sum_mask_m, min=1e-9)
                image_sum_mask_m = image_mask_expanded_m.sum(axis=1)
                image_sum_mask_m = torch.clamp(image_sum_mask_m, min=1e-9)
                mean_audio_feat_m = audio_sum_embeddings_m / audio_sum_mask_m # torch.Size([1, audio_width]
                mean_text_feat_m = text_sum_embeddings_m / text_sum_mask_m # torch.Size([1, text_width]
                mean_image_feat_m = image_sum_embeddings_m / image_sum_mask_m

                image_feat_m = F.normalize(self.vision_proj_m(mean_image_feat_m),dim=-1)
                image_feat_m_l = F.normalize(self.vision_proj_m(image_embeds_m),dim=-1)  
                # image_feat_m_l = self.patch_pooling(image_feat_m_l) # pooling for image patches
                image_feat_all = torch.cat([image_feat_m.t(),self.image_queue.clone().detach()],dim=1)

                audio_feat_m = F.normalize(self.audio_proj_m(mean_audio_feat_m),dim=-1)
                audio_feat_m_l = F.normalize(self.audio_proj_m(audio_embeds_m), dim=-1)
                audio_feat_all = torch.cat([audio_feat_m.t(),self.audio_queue.clone().detach()],dim=1)
    
                text_feat_m = F.normalize(self.text_proj_m(mean_text_feat_m),dim=-1)
                text_feat_m_l = F.normalize(self.text_proj_m(text_embeds_m),dim=-1) 
                text_feat_all = torch.cat([text_feat_m.t(),self.text_queue.clone().detach()],dim=1)

                sim_i2a_m = image_feat_m @ audio_feat_all / self.temp 
                sim_a2i_m = audio_feat_m @ image_feat_all / self.temp     

                sim_i2t_m = image_feat_m @ text_feat_all / self.temp 
                sim_t2i_m = text_feat_m @ image_feat_all / self.temp     

                sim_a2t_m = audio_feat_m @ text_feat_all / self.temp
                sim_t2a_m = text_feat_m @ audio_feat_all / self.temp

                sim_targets_ia = torch.zeros(sim_i2a_m.size()).to(video_input.device)
                sim_targets_ia.fill_diagonal_(1)
                sim_i2a_targets = alpha * F.softmax(sim_i2a_m, dim=1) + (1 - alpha) * sim_targets_ia
                sim_a2i_targets = alpha * F.softmax(sim_a2i_m, dim=1) + (1 - alpha) * sim_targets_ia     

                sim_targets_it = torch.zeros(sim_i2t_m.size()).to(video_input.device)
                sim_targets_it.fill_diagonal_(1)
                sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets_it
                sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets_it     

                sim_targets_at = torch.zeros(sim_a2t_m.size()).to(video_input.device)
                sim_targets_at.fill_diagonal_(1)
                sim_a2t_targets = alpha * F.softmax(sim_a2t_m, dim=1) + (1 - alpha) * sim_targets_at
                sim_t2a_targets = alpha * F.softmax(sim_t2a_m, dim=1) + (1 - alpha) * sim_targets_at
            
            sim_i2a = image_feat @ audio_feat_all / self.temp 
            sim_a2i = audio_feat @ image_feat_all / self.temp

            sim_i2t = image_feat @ text_feat_all / self.temp
            sim_t2i = text_feat @ image_feat_all / self.temp 

            sim_a2t = audio_feat @ text_feat_all / self.temp
            sim_t2a = text_feat @ audio_feat_all / self.temp

            loss_i2a = -torch.sum(F.log_softmax(sim_i2a, dim=1)*sim_i2a_targets,dim=1).mean()
            loss_a2i = -torch.sum(F.log_softmax(sim_a2i, dim=1)*sim_a2i_targets,dim=1).mean() 

            loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1)*sim_i2t_targets,dim=1).mean()
            loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1)*sim_t2i_targets,dim=1).mean() 

            loss_a2t = -torch.sum(F.log_softmax(sim_a2t, dim=1)*sim_a2t_targets,dim=1).mean()
            loss_t2a = -torch.sum(F.log_softmax(sim_t2a, dim=1)*sim_t2a_targets,dim=1).mean() 

            # acformer: add inMod g2l loss
            loss_i2i_inmod_l = self.in_batch_g2l_loss(image_feat_m_l, image_feat, self.temp)
            loss_a2a_inmod_l = self.in_batch_g2l_loss(audio_feat_m_l, audio_feat, self.temp)
            loss_t2t_inmod_l = self.in_batch_g2l_loss(text_feat_m_l, text_feat, self.temp)

            # acformer: add in-modality g2g loss
            sim_i2i = image_feat @ image_feat_all / self.temp
            sim_a2a = audio_feat @ audio_feat_all / self.temp
            sim_t2t = text_feat @ text_feat_all / self.temp

            loss_i2i = -torch.sum(F.log_softmax(sim_i2i, dim=1)*sim_targets_ia,dim=1).mean()
            loss_a2a = -torch.sum(F.log_softmax(sim_a2a, dim=1)*sim_targets_it,dim=1).mean()
            loss_t2t = -torch.sum(F.log_softmax(sim_t2t, dim=1)*sim_targets_at,dim=1).mean()
            # compute multimodal vision-audio-text loss
            loss_vat = (loss_i2i_inmod_l + loss_a2a_inmod_l + loss_t2t_inmod_l  + loss_i2a + loss_a2i + loss_i2t + loss_t2i + loss_a2t + loss_t2a + loss_i2i + loss_a2a + loss_t2t) / 12.0

            #self._dequeue_and_enqueue(image_feat_m, audio_feat_m, text_feat_m)

            return loss_vat 
        else:
            return image_feat, audio_feat, text_feat
    
    @torch.no_grad()    
    def copy_params(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient    

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, audio_feat, text_feat):
        # gather keys before updating queue
        #image_feats = concat_all_gather(image_feat)
        #audio_feats = concat_all_gather(audio_feat)
        #text_feats = concat_all_gather(text_feat)
        image_feats = image_feat
        audio_feats = audio_feat
        text_feats = text_feat
 
        batch_size = image_feats.shape[0]
 
        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity
 
        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.audio_queue[:, ptr:ptr + batch_size] = audio_feats.T        
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer
 
        self.queue_ptr[0] = ptr 
 
    @torch.no_grad()        
    def _momentum_update(self):
        for model_pair in self.model_pairs:           
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)
    

    def patch_pooling(self, x):
        batch_size, seq_length, dim = x.size()
        b1 = int(np.sqrt(seq_length))
        x = x.reshape(batch_size, b1, b1, dim)
        x = x.permute(0,3,1,2)
        c1 = int(np.sqrt(b1))
        x = F.avg_pool2d(x, c1, stride=c1)
        x = x.permute(0,2,3,1).reshape(batch_size, c1*c1, dim)
        return x

    def in_batch_g2l_loss(self, l, m, temp, attention_mask=None):
        m = m.unsqueeze(1)
        N, n_locals, dim = l.size()
        l_n = l.reshape(-1, dim) # (N * n_locals) * d
        m_n = m.reshape(-1, dim) # N * d

        # Inner product for positive samples. Outer product for negative. We need to do it this way
        # for the multiclass loss. For the outer product, we want a N x N x n_locals x 1 tensor.
        u_p = torch.matmul(l, m.permute(0,2,1)).unsqueeze(2) / temp # N * n_locals * 1 * 1
        
        # if l comes from text, then attention_mask is not None
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(2).unsqueeze(3)
            u_p = (temp_mask * u_p) + (10000. * (1-temp_mask))
        
        u_n = torch.mm(m_n, l_n.t()) / temp
        u_n = u_n.reshape(N, 1, N, n_locals).permute(0, 2, 3, 1) # N x N x n_locals x 1

        # We need to mask the diagonal part of the negative tensor.
        mask = torch.eye(N)[:, :, None, None].to(l.device) # N*N*1*1
        n_mask = 1 - mask

        # Masking is done by shifting the diagonal before exp.
        u_n = (n_mask * u_n) - (10000. * (1 - n_mask))  # mask out "self" examples
        # if l comes from test, we mask out the padding tokens
        if attention_mask is not None:
            temp_mask = attention_mask.unsqueeze(0).unsqueeze(3).expand(N, -1, -1, -1)
            u_n = (temp_mask * u_n) - (10000. * (1-temp_mask))

        u_n = u_n.reshape(N, N * n_locals, 1).unsqueeze(dim=1).expand(-1, n_locals, -1, -1)

        # Since this is multiclass, we concat the positive along the class dimension before performing log softmax.
        pred_lgt = torch.cat([u_p, u_n], dim=2)
        pred_log = F.log_softmax(pred_lgt, dim=2)

        # The positive score is the first element of the log softmax.
        if attention_mask is not None:
            loss = (torch.sum(-pred_log[:, :, 0].squeeze(), dim=1) / torch.sum(attention_mask, dim=1)).mean()
        else:
            loss = -pred_log[:, :, 0].mean()

        return loss


def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)
 
    output = torch.cat(tensors_gather, dim=0)
    return output