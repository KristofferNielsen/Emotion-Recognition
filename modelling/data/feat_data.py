import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset
from utils.read_data import *
import time
from torch.nn.utils.rnn import pad_sequence

class Data_Feat(Dataset):
    def __init__(self, root, names, labels,type):
        # analyze path
        self.names = names
        self.labels = labels
        #self.ltpp = LTPP()
        root = root
        self.audio_root = root + 'audio_wavlm/'
        self.text_root = root + 'text_bloom/'
        self.video_root = root + 'video/'
        #print(f'audio feature root: {audio_root}')

        # analyze params
        self.feat_scale = 1 # 特征预压缩
        audios, self.adim = func_read_multiprocess(self.audio_root, self.names, read_type='feat')
        texts,  self.tdim = func_read_multiprocess(self.text_root,  self.names, read_type='feat')
        videos, self.vdim = func_read_multiprocess(self.video_root, self.names, read_type='feat')
        # read batch (reduce collater durations)
        # step1: pre-compress features
        audios, texts, videos = feature_scale_compress(audios, texts, videos, self.feat_scale)
        # step2: align to batch (only utt type)
        if type == "1d":
            audios, texts, videos = align_to_utt(audios, texts, videos)
        #elif type == "2d":
        #    audios, texts, videos = pad_to_maxlen_pre_modality(audios, texts, videos)
        self.audios, self.texts, self.videos = audios, texts, videos

    def __len__(self):
        return len(self.names)

    def __getitem__(self, index):
        if "iemocap" in self.audio_root:
            instance = dict(
            audio = self.audios[index],
            text  = self.texts[index],
            video = self.videos[index],
            emo   = self.labels[index][0],
            val   = self.labels[index][1],
            name  = self.names[index],
        )
        else:
            instance = dict(
                audio = self.audios[index],
                text  = self.texts[index],
                video = self.videos[index],
                emo   = self.labels[index]['emo'],
                val   = self.labels[index]['val'],
                name  = self.names[index],
            )
        return instance

    def collater(self, instances):

        audios = [torch.FloatTensor(instance['audio']) for instance in instances]
        texts  = [torch.FloatTensor(instance['text']) for instance in instances]
        videos = [torch.FloatTensor(instance['video']) for instance in instances]

        # Pad sequences per batch
        padded_audios = pad_sequence(audios, batch_first=True, padding_value=0)
        padded_texts = pad_sequence(texts, batch_first=True, padding_value=0)
        padded_videos = pad_sequence(videos, batch_first=True, padding_value=0)

        audio_masks = (padded_audios.abs().sum(dim=-1) != 0).float()
        text_masks = (padded_texts.abs().sum(dim=-1) != 0).float()
        video_masks = (padded_videos.abs().sum(dim=-1) != 0).float()

        batch = dict(
            audios=padded_audios,
            texts=padded_texts,
            videos=padded_videos,
            audio_masks=audio_masks,
            text_masks=text_masks,
            video_masks=video_masks,
        )
        
        emos  = torch.LongTensor([instance['emo']  for instance in instances])
        vals  = torch.FloatTensor([instance['val']  for instance in instances])
        return batch['audios'], batch['texts'], batch['videos'], batch['audio_masks'], batch['text_masks'], batch['video_masks'], emos, vals
