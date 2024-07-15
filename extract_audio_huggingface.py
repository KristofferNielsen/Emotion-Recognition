import os
import math
import time
import glob
import torch
import argparse
import numpy as np
import soundfile as sf

# import config
import sys
sys.path.append('../../')

from transformers import AutoModel
from transformers import Wav2Vec2FeatureExtractor

#WAV2VEC2_LARGE = 'wav2vec2-large-960h' # https://huggingface.co/facebook/wav2vec2-large-960h

## Target: avoid too long inputs
# input_values: [1, wavlen], output: [bsize, maxlen]
def split_into_batch(input_values, maxlen=16000*10):
    if len(input_values[0]) <= maxlen:
        return input_values
    
    bs, wavlen = input_values.shape
    assert bs == 1
    tgtlen = math.ceil(wavlen / maxlen) * maxlen
    batches = torch.zeros((1, tgtlen))
    batches[:, :wavlen] = input_values
    batches = batches.view(-1, maxlen)
    return batches

def extract(audio_files, save_dir):

    start_time = time.time()

    # load model

    model = AutoModel.from_pretrained('facebook/wav2vec2-xls-r-2b')
    feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('facebook/wav2vec2-xls-r-2b')

    device = torch.device(f'cuda:{0}')
    model.to(device)
    model.eval()
    # iterate audios
    for idx, audio_file in enumerate(audio_files, 1):
        file_name = os.path.basename(audio_file)
        audio_name = file_name[:-4]
        print(f'Processing "{audio_file}" ({idx}/{len(audio_files)})...')
        ## process for too short ones
        samples, sr = sf.read('/work3/s194644/data/mer/audio_pre/' + audio_file)
        assert sr == 16000, 'currently, we only test on 16k audio'
        
        ## model inference
        with torch.no_grad():
            layer_ids = [-4, -3, -2, -1]
            input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values # [1, wavlen]
            input_values = split_into_batch(input_values) # [bsize, maxlen=10*16000]
            input_values = input_values.to(device)
            hidden_states = model(input_values, output_hidden_states=True).hidden_states # tuple of (B, T, D)
            feature = torch.stack(hidden_states)[layer_ids].sum(dim=0)  # (B, T, D) # -> compress waveform channel
            bsize, segnum, featdim = feature.shape
            feature = feature.view(-1, featdim).detach().squeeze().cpu().numpy() # (B*T, D)

        ## store values
        save_file = os.path.join(save_dir, f'{audio_name}.npy')
        np.save(save_file, feature)

    end_time = time.time()
    print(f'Total time used: {end_time - start_time:.1f}s.')


if __name__ == '__main__':

    audio_files = os.listdir("/work3/s194644/data/mer/audio_pre/")
    print(f'Find total "{len(audio_files)}" audio files.')

    save_dir = "/work3/s194644/data/mer/audio_xlsr"
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    
    
    # extract features
    extract(audio_files, save_dir)
