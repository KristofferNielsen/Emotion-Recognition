import torch
from torch import nn
import torch
import torch.nn.functional as F
from utils.modules import ConvBlock, Seq

class ModelCMHA(nn.Module):
    def __init__(self, input_dims, d_model, nhead, num_classes, dropout=0.1):
        super(ModelCMHA, self).__init__()
        self.audio_encoder = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.text_encoder = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.image_encoder =  Seq(d_model, d_model,nhead, dropout_prob=dropout)

        self.cross_audio1 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.cross_audio2 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.cross_text1 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.cross_text2 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.cross_image1 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.cross_image2 = Seq(d_model, d_model,nhead, dropout_prob=dropout)

        self.audio_attention1 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.audio_attention2 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.text_attention1 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.text_attention2 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.image_attention1 = Seq(d_model, d_model,nhead, dropout_prob=dropout)  
        self.image_attention2 = Seq(d_model, d_model,nhead, dropout_prob=dropout)

        self.fusion1 = Seq(d_model, d_model,nhead, dropout_prob=dropout)  
        self.fusion2 = Seq(d_model, d_model,nhead, dropout_prob=dropout)
        self.fusion3 = Seq(d_model, d_model,nhead, dropout_prob=dropout)

        self.dropout = nn.Dropout(dropout)
        self.proj1 = nn.Linear(d_model*3, d_model*3)
        self.proj2 = nn.Linear(d_model*3, d_model*3)
        self.out_layer = nn.Linear(d_model*3, num_classes)
        self.out_layer1 = nn.Linear(d_model*3, 1)
        
        self.audio_conv = ConvBlock(in_channels=input_dims[0], out_channels=d_model, dropout_prob=dropout)
        self.text_conv = ConvBlock(in_channels=input_dims[1], out_channels=d_model, dropout_prob=dropout)
        self.visual_conv = ConvBlock(in_channels=input_dims[2], out_channels=d_model, dropout_prob=dropout)
        
    def forward(self, mods):
        audio = self.audio_conv(mods[0].permute(0,2,1)).permute(0,2,1)
        text = self.text_conv(mods[1].permute(0,2,1)).permute(0,2,1)
        image = self.visual_conv(mods[2].permute(0,2,1)).permute(0,2,1)

        audio = self.audio_encoder(audio,audio)
        text = self.text_encoder(text,text)
        image = self.image_encoder(image,image)

        audio1 = self.cross_audio1(audio, text)
        audio2 = self.cross_audio2(audio, image)
        text1 = self.cross_text1(text, audio)
        text2 = self.cross_text2(text, image)
        image1 = self.cross_image1(image, audio)
        image2 = self.cross_image2(image, text)

        audio1 = self.audio_attention1(audio1, image) 
        text1 = self.text_attention1(text1, image)
        cat = torch.cat((audio1, text1), dim=1)

        audio2 = self.audio_attention2(audio2, text)
        image1 = self.image_attention1(image1, text)
        cat1 = torch.cat((audio2, image1), dim=1)

        text2 = self.text_attention2(text2, audio)
        image2 = self.image_attention2(image2, audio)
        cat2 = torch.cat((text2, image2), dim=1)

        z1 = self.fusion1(cat, cat)
        z2 = self.fusion2(cat1, cat1)
        z3 = self.fusion3(cat2, cat2)
        # Concatenate the output of the three encoder
        z = torch.cat((z1[:,0,:], z2[:,0,:], z3[:,0,:]), dim=-1)
        last_hs_proj = self.proj2(self.dropout(F.relu(self.proj1(z))))
        
        output = self.out_layer(last_hs_proj)
        output1 = self.out_layer1(last_hs_proj)
        return output, output1
    
    
