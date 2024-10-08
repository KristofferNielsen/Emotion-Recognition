import torch
from torch import nn
import torch
import torch.nn.functional as F
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

class ModelCM1d(nn.Module):
    def __init__(self, input_dims, d_model, nhead, num_classes, dropout=0.1):
        super(ModelCM1d, self).__init__()
        self.audio_encoder = MLPEncoder(input_dims[0], d_model, dropout)
        self.text_encoder  = MLPEncoder(input_dims[1],  d_model, dropout)
        self.video_encoder = MLPEncoder(input_dims[2], d_model, dropout)

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

        self.conv_at1 = nn.Conv1d(d_model*2,d_model,1)
        self.conv_at2 = nn.Conv1d(d_model*2,d_model,1)
        self.conv_av1 = nn.Conv1d(d_model*2,d_model,1)
        self.conv_av2 = nn.Conv1d(d_model*2,d_model,1)
        self.conv_tv1 = nn.Conv1d(d_model*2,d_model,1)
        self.conv_tv2 = nn.Conv1d(d_model*2,d_model,1)

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(d_model*3, d_model*3)
        self.activation_fn = nn.ReLU()
        self.out_proj1 = nn.Linear(d_model*3, num_classes)
        self.out_proj2 = nn.Linear(d_model*3, 1)
        
    def forward(self, mods):
        audio = self.audio_encoder(mods[0]).unsqueeze(1)
        text = self.text_encoder(mods[1]).unsqueeze(1)
        image = self.video_encoder(mods[2]).unsqueeze(1)

        audio1 = self.cross_audio1(audio, text)
        audio2 = self.cross_audio2(audio, image)
        text1 = self.cross_text1(text, audio)
        text2 = self.cross_text2(text, image)
        image1 = self.cross_image1(image, audio)
        image2 = self.cross_image2(image, text)

        cat = torch.cat((audio1, text1), dim=2)
        conv = self.conv_at1(cat.permute(0,2,1)).permute(0,2,1)
        audio1 = self.audio_attention1(conv, image) 
        text1 = self.text_attention1(image, conv)
        cat = torch.cat((audio1, text1), dim=2)
        z1 = self.conv_at2(cat.permute(0,2,1)).permute(0,2,1).squeeze(1)

        cat1 = torch.cat((audio2, image1), dim=2)
        conv1 = self.conv_av1(cat1.permute(0,2,1)).permute(0,2,1)
        audio2 = self.audio_attention2(conv1, text)
        image1 = self.image_attention1(text, conv1)
        cat1 = torch.cat((audio2, image1), dim=2)
        z2 = self.conv_av2(cat1.permute(0,2,1)).permute(0,2,1).squeeze(1)

        cat2 = torch.cat((text2, image2), dim=2)
        conv2 = self.conv_tv1(cat2.permute(0,2,1)).permute(0,2,1)
        text2 = self.text_attention2(conv2, audio)
        image2 = self.image_attention2(audio, conv2)
        cat2 = torch.cat((text2, image2), dim=2)
        z3 = self.conv_tv2(cat2.permute(0,2,1)).permute(0,2,1).squeeze(1)

        # Concatenate the output of the three encoder
        fused_feat = torch.cat((z1, z2, z3), dim=1)
        x = self.dense(fused_feat)
        x = self.activation_fn(x)
        x = self.dropout(x)
        x1 = self.out_proj1(x)
        x2  = self.out_proj2(x)
        return x1, x2
    
    
