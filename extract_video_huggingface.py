# *_*coding:utf-8 *_*
import os
import cv2
import math
import argparse
import numpy as np
from PIL import Image

import torch 
import timm # pip install timm==0.9.7
from transformers import AutoModel, AutoFeatureExtractor, AutoImageProcessor

# import config
import sys

##################### Pretrained models #####################
CLIP_VIT_LARGE = 'clip-vit-large-patch14' # https://huggingface.co/openai/clip-vit-large-patch14


def func_opencv_to_image(img):
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    return img

def func_opencv_to_numpy(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def func_read_frames(face_dir, vid):
    avi_path = os.path.join(face_dir, vid)
    #print(avi_path)
    #assert os.path.exists(avi_path), f'Error: {vid} does not have {vid}.avi!'
    
    cap = cv2.VideoCapture(avi_path)
    frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)

    cap.release()
    frames = np.array(frames)
    
    return frames

# 策略3：相比于上面采样更加均匀 [将videomae替换并重新测试]
def resample_frames_uniform(frames, nframe=16):
    vlen = len(frames)
    start, end = 0, vlen
    
    n_frms_update = min(nframe, vlen) # for vlen < n_frms, only read vlen
    indices = np.arange(start, end, vlen / n_frms_update).astype(int).tolist()
    
    # whether compress into 'n_frms'
    while len(indices) < nframe:
        indices.append(indices[-1])
    indices = indices[:nframe]
    assert len(indices) == nframe, f'{indices}, {vlen}, {nframe}'
    return frames[indices]
    
def split_into_batch(inputs, bsize=32):
    batches = []
    for ii in range(math.ceil(len(inputs)/bsize)):
        batch = inputs[ii*bsize:(ii+1)*bsize]
        batches.append(batch)
    return batches


if __name__ == '__main__':

    # gain save_dir
    save_dir = 'mer_video_2d'
    #save_dir1 = 'mer_video_1d'
    if not os.path.exists(save_dir): os.makedirs(save_dir)
    #if not os.path.exists(save_dir1): os.makedirs(save_dir1)

    # load model
    #model_dir = os.path.join(config.PATH_TO_PRETRAINED_MODELS, f'transformers/{params.model_name}')
    model = AutoModel.from_pretrained('openai/clip-vit-large-patch14')
    processor  = AutoFeatureExtractor.from_pretrained('openai/clip-vit-large-patch14')
    
    # 有 gpu 才会放在cuda上
    torch.cuda.set_device(0)
    model.cuda()
    model.eval()

    # extract embedding video by video
    vids = os.listdir('mer2024/video/')
    EMBEDDING_DIM = -1
    print(f'Find total "{len(vids)}" videos.')
    for i, vid in enumerate(vids, 1):
        file_name = os.path.basename(vid)
        vid_name = file_name[:-4]
        print(f"Processing video '{vid_name}' ({i}/{len(vids)})...")
        # forward process [different model has its unique mode, it is hard to unify them as one process]
        # => split into batch to reduce memory usage
        with torch.no_grad():
            frames = func_read_frames('mer2024/video/', vid)
            frames = [func_opencv_to_image(frame) for frame in frames]
            inputs = processor(images=frames, return_tensors="pt")['pixel_values']
            inputs = inputs.to("cuda")
            batches = split_into_batch(inputs, bsize=32)
            embeddings = []
            for batch in batches:
                embeddings.append(model.get_image_features(batch)) # [58, 768]
            embeddings = torch.cat(embeddings, axis=0) # [frames_num, 768]

    
        embeddings = embeddings.detach().squeeze().cpu().numpy()
        EMBEDDING_DIM = max(EMBEDDING_DIM, np.shape(embeddings)[-1])

        # save into npy

        save_file = os.path.join(save_dir, f'{vid_name}.npy')
        embeddings = np.array(embeddings).squeeze()
        if len(embeddings) == 0:
            embeddings = np.zeros((1, EMBEDDING_DIM))
        elif len(embeddings.shape) == 1:
            embeddings = embeddings[np.newaxis, :]
        np.save(save_file, embeddings)

