from collections import defaultdict
import os
import pandas
from datasets import load_dataset, load_metric, Dataset, DatasetDict
import soundfile as sf
import math
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
import pickle
import hydra
import pandas as pd
import cv2


RAW_PATH = "IEMOCAP/IEMOCAP_full_release/"  # path to raw data
#Get data from folder


@hydra.main(config_path="../../conf", config_name="config.yaml", version_base=None)
def process_data(path) -> None:
    all_data = []
    for session in ['Session1', 'Session2', 'Session3', 'Session4', 'Session5']:
        for transcript in os.scandir(path +session+'/dialog/EmoEvaluation/'):
            if transcript.is_dir(): continue
            transcript_name = transcript.path.split('/')[-1]
            if transcript_name.startswith("."): continue
            recording_session = transcript_name.replace('.txt', '')
            with open(transcript, 'r') as f:
                line = f.readline()
                while line:
                        if line.startswith("["):
                            chunks = line.split("\t")
                            all_data.append((chunks[1],chunks[0],chunks[2],path + session, recording_session))
                        line = f.readline()
    #save the data in a csv file
    df = pd.DataFrame(all_data, columns=['Titel', 'Time', 'Emotion','Path', 'Recording'])
    df.to_csv('modelling\data\Labels\IEMOCAP.csv', index=False)
    for sample,time,path,recording in df[['Titel','Time', 'Path','Recording']].values:
        audiofile = path + '/sentences/wav/' + recording + '/' + sample + ".wav"
        #save the audio file
        audio, samplerate = sf.read(audiofile, always_2d=False, dtype='int16')
        sf.write('modelling\data\Audio\\' + sample + ".wav", audio, samplerate)

        textfile = path + '/dialog/transcriptions/' + recording + ".txt"
        with open(textfile, 'r') as f:
            line = f.readline()
            while line:
                chunks = line.split("\t")
                for chunk in chunks:
                    #split chunk in 3 by the first two spaces
                    chunk = chunk.split(" ", 2)
                    if chunk[0] == sample:
                        #save chunk[2]
                        with open('modelling\data\Text\\' + sample + ".txt", 'w') as q:
                            q.write(chunk[2])
                line = f.readline()

        videofile = path + '/dialog/avi/DivX/' + recording + ".avi"
        start,end = time[1:-1].split('-')
        start = np.floor(float(start))
        end = np.ceil(float(end))
        cap = cv2.VideoCapture(videofile)
        fps = cap.get(cv2.CAP_PROP_FPS)
        #saving frames in new video
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        writer = cv2.VideoWriter(f"modelling\data\Videos\{sample}.avi", fourcc, fps, (360, 260))
        cap.set(cv2.CAP_PROP_POS_MSEC, start*1000)
        f = 0
        while f < (end-start)*fps:
            f += 1
            ret, frame = cap.read()
            if ret:
                if sample[5] == sample[-4]:
                    frame = frame[110:370, :360]
                else:
                    frame = frame[110:370, 360:]
                writer.write(frame)
        writer.release()
        cap.release() 

if __name__ == "__main__":
    process_data("IEMOCAP/IEMOCAP_full_release/")

