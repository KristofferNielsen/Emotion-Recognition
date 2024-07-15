# *_*coding:utf-8 *_*
import os
import time
import argparse
import numpy as np
import pandas as pd

import torch
from transformers import AutoModel, BertTokenizer, AutoTokenizer # version: 4.5.1, pip install transformers
#from transformers import GPT2Tokenizer, GPT2Model, AutoModelForCausalLM

# local folder
import sys
sys.path.append('../../')

##################### English #####################
ROBERTA_LARGE = 'roberta-large'

################################################################
# 自动删除无意义token对应的特征
def find_start_end_pos(tokenizer):
    sentence = '今天天气真好' # 句子中没有空格
    input_ids = tokenizer(sentence, return_tensors='pt')['input_ids'][0]
    start, end = None, None

    # find start, must in range [0, 1, 2]
    for start in range(0, 3, 1):
        # 因为decode有时会出现空格，因此我们显示的时候把这部分信息去掉看看
        outputs = tokenizer.decode(input_ids[start:]).replace(' ', '')
        if outputs == sentence:
            print (f'start: {start};  end: {end}')
            return start, None

        if outputs.startswith(sentence):
            break
   
    # find end, must in range [-1, -2]
    for end in range(-1, -3, -1):
        outputs = tokenizer.decode(input_ids[start:end]).replace(' ', '')
        if outputs == sentence:
            break
    
    assert tokenizer.decode(input_ids[start:end]).replace(' ', '') == sentence
    print (f'start: {start};  end: {end}')
    return start, end


# 找到 batch_pos and feature_dim
def find_batchpos_embdim(tokenizer, model, gpu):
    sentence = '今天天气真好'
    inputs = tokenizer(sentence, return_tensors='pt')
    if gpu != -1: inputs = inputs.to('cuda')

    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
        outputs = torch.stack(outputs)[[-1]].sum(dim=0) # sum => [batch, T, D=768]
        outputs = outputs.cpu().numpy() # (B, T, D) or (T, B, D)
        batch_pos = None
        if outputs.shape[0] == 1:
            batch_pos = 0
        if outputs.shape[1] == 1:
            batch_pos = 1
        assert batch_pos in [0, 1]
        feature_dim = outputs.shape[2]
    print (f'batch_pos:{batch_pos}, feature_dim:{feature_dim}')
    return batch_pos, feature_dim


# main process
def extract_embedding():
    start_time = time.time()

    # save last four layers
    layer_ids = [-4, -3, -2, -1]

    save_dir = "data/mer/text_english_roberta"
    if not os.path.exists(save_dir): os.makedirs(save_dir)

    # load model and tokenizer: offline mode (load cached files) # 函数都一样，但是有些位置的参数就不好压缩

    #model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-large')
    #tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large', use_fast=False)

    model = AutoModel.from_pretrained('FacebookAI/roberta-large')
    tokenizer = AutoTokenizer.from_pretrained('FacebookAI/roberta-large', use_fast=False)
   
    torch.cuda.set_device(0)
    model.cuda()
    model.eval()

    print('Calculate embeddings...')
    start, end = find_start_end_pos(tokenizer) # only preserve [start:end+1] tokens
    batch_pos, feature_dim = find_batchpos_embdim(tokenizer, model, 0) # find batch pos

    df = pd.read_csv('data/mer/label-transcription.csv')
    for idx, row in df.iterrows():
        name = row['name']
        sentence = row['english']
        # --------------------------------------------------
        print(f'Processing {name} ({idx}/{len(df)})...')

        # extract embedding from sentences
        embeddings = []
        if pd.isna(sentence) == False and len(sentence) > 0:
            inputs = tokenizer(sentence, return_tensors='pt')
            inputs = inputs.to('cuda')
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True).hidden_states # for new version 4.5.1
                outputs = torch.stack(outputs)[layer_ids].sum(dim=0) # sum => [batch, T, D=768]
                outputs = outputs.cpu().numpy() # (B, T, D)
                if batch_pos == 0:
                    embeddings = outputs[0, start:end]
                elif batch_pos == 1:
                    embeddings = outputs[start:end, 0]

        # align with label timestamp and write csv file
        print (f'feature dimension: {feature_dim}')
        csv_file = os.path.join(save_dir, f'{name}.npy')

        embeddings = np.array(embeddings).squeeze()
        if len(embeddings) == 0:
            embeddings = np.zeros((1, feature_dim))
        elif len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
        np.save(csv_file, embeddings)


    end_time = time.time()
    print(f'Total {len(df)} files done! Time used: {end_time - start_time:.1f}s.')



if __name__ == '__main__':

    extract_embedding()
