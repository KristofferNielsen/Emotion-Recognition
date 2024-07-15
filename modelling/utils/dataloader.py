#create dataloader that loads the labels,videos,text and audio for each sample
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import cv2
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
#from transformers import Wav2Vec2FeatureExtractor, AutoTokenizer, VideoMAEImageProcessor
import librosa
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.sampler import SubsetRandomSampler
from data import get_datasets
from torch.utils.data import ConcatDataset
import random
from collections import defaultdict
#create dataloader that loads the labels,videos,text and audio for each sample

class dataloader(Dataset):
    def __init__(self, root, num_class=4,train=True,val=False):
        self.root = root
        self.data = pd.read_csv(root + 'modelling/data/Labels/IEMOCAP.csv')
        self.data = self.data[self.data['Titel'] != 'Ses03M_script03_2_M003']
        if train:
            self.data = self.data[self.data['Session'].isin(['Session2','Session3', 'Session4'])]
        elif val:
            self.data = self.data[self.data['Session'].isin(['Session1'])]
        else:
            self.data = self.data[self.data['Session'].isin(['Session5'])]
        if num_class == 4:
            self.data = self.data[self.data['Emotion'].isin(['ang', 'hap', 'sad', 'neu','exc'])]
            self.emotion_dict = {'ang': 0, 'hap': 1,'exc':1, 'sad': 2, 'neu': 3}
        else:
            self.data = self.data[self.data['Emotion'].isin(['ang', 'hap', 'sad', 'neu', 'exc', 'fru', 'fea', 'sur'])]
            self.emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'exc': 4, 'fru': 5, 'fea': 6, 'sur': 7}
        #self.audio = root + 'modelling/data/Audio/'
        #self.text = root + 'modelling/data/Text/'
        #self.video = root + 'modelling/data/Videos/'
        self.data['Emotion_cont'] = self.data['Emotion_cont'].apply(lambda x: x.split('[')[1].split(',')[0])
        self.data['Emotion_cont'] =self.data['Emotion_cont'].apply(lambda x: float(x))
        self.audio = root + 'audio_1d/'
        self.text = root + 'text_1d/'
        self.video = root + 'video_1d/'
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        audiofile = np.load(self.audio + sample['Titel'] + ".wav.npy")
        textfile = np.load(self.text + sample['Titel'] + ".npy")    
        videofile = np.load(self.video + sample['Titel'] + ".avi.npy")

        emotion = torch.tensor(self.emotion_dict[sample['Emotion']])
        valence = torch.tensor(sample['Emotion_cont'])
        return torch.tensor(audiofile), torch.tensor(textfile), torch.tensor(videofile),emotion, valence
    
    def __len__(self):
        return len(self.data)

class LTPP(nn.Module):
    def __init__(self, layers=None):
        super(LTPP, self).__init__()
        if layers is None:
            layers = [1, 2, 4, 8, 16]
        self.layers = layers

    def forward(self, x):
        T = x.shape[0]
        final = []
        for l in self.layers:
            t = nn.AdaptiveAvgPool1d(l)(x.transpose(0, 1))
            final.append(t.transpose(0, 1))
        t = F.max_pool1d(x.transpose(0, 1), T)
        final.append(t.transpose(0, 1))

        t = torch.cat(final, dim=0)
        return t

class dataloader1(Dataset):
    def __init__(self, root, num_class=4,train=True,val=False):
        self.root = root
        self.data = pd.read_csv(root + 'modelling/data/Labels/IEMOCAP.csv')
        self.data = self.data[self.data['Titel'] != 'Ses03M_script03_2_M003']
        if train:
            self.data = self.data[self.data['Session'].isin(['Session2','Session3', 'Session4'])]
        elif val:
            self.data = self.data[self.data['Session'].isin(['Session1'])]
        else:
            self.data = self.data[self.data['Session'].isin(['Session5'])]
        if num_class == 4:
            self.data = self.data[self.data['Emotion'].isin(['ang', 'hap', 'sad', 'neu','exc'])]
            self.emotion_dict = {'ang': 0, 'hap': 1,'exc':1, 'sad': 2, 'neu': 3}
        else:
            self.data = self.data[self.data['Emotion'].isin(['ang', 'hap', 'sad', 'neu', 'exc', 'fru', 'fea', 'sur'])]
            self.emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'exc': 4, 'fru': 5, 'fea': 6, 'sur': 7}
        self.audio = root + 'audio_2d/'
        self.text = root + 'text_2d/'
        self.video = root + 'video_2d/'
        self.data['Emotion_cont'] = self.data['Emotion_cont'].apply(lambda x: x.split('[')[1].split(',')[0])
        self.data['Emotion_cont'] =self.data['Emotion_cont'].apply(lambda x: float(x))
        self.ltpp = LTPP()
    
    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        audiofile = np.load(self.audio + sample['Titel'] + ".wav.npy")
        textfile = np.load(self.text + sample['Titel'] + ".npy")    
        videofile = np.load(self.video + sample['Titel'] + ".avi.npy")
        videofile = torch.tensor(videofile)
        textfile= torch.tensor(textfile)
        audiofile= torch.tensor(audiofile)

        if videofile.shape[0] < 16:
            videofile = torch.cat((videofile, torch.zeros(16 - videofile.shape[0], videofile.shape[1])), 0)
        if textfile.shape[0] < 16:
            textfile = torch.cat((textfile, torch.zeros(16 - textfile.shape[0], textfile.shape[1])), 0)
        if audiofile.shape[0] < 16:
            audiofile = torch.cat((audiofile, torch.zeros(16 - audiofile.shape[0], audiofile.shape[1])), 0)
        audiofile = self.ltpp(audiofile)
        textfile = self.ltpp(textfile)
        videofile = self.ltpp(videofile)
        emotion = torch.tensor(self.emotion_dict[sample['Emotion']])
        valence = torch.tensor(sample['Emotion_cont'])
        return audiofile,textfile, videofile, emotion, valence
    
    def __len__(self):
        return len(self.data)

class DM(LightningDataModule):
    def __init__(self,dataload, data_path,  batch_size, num_class):
        super().__init__()
        #load the dataset
        self.num_class = num_class
        #dataset = dataloader(data_path, self.num_class,train=True)
        if dataload == "dataloader1":
            self.data_train = dataloader1(data_path, self.num_class,train=True,val=False)
            self.data_val = dataloader1(data_path, self.num_class,train=False,val=True)
            self.data_test = dataloader1(data_path, self.num_class,train=False)
        else: 
            self.data_train = dataloader(data_path, self.num_class,train=True,val=False)
            self.data_val = dataloader(data_path, self.num_class,train=False,val=True)
            self.data_test = dataloader(data_path, self.num_class,train=False)
        #self.data_train, self.data_val = torch.utils.data.random_split(dataset, [0.75, 0.25])
        self.batch_size = batch_size

    def train_dataloader(self):
        return DataLoader(self.data_train, num_workers=4, batch_size=self.batch_size, shuffle=True, drop_last=True)#,collate_fn=collate_fn)

    def val_dataloader(self):
        return DataLoader(self.data_val, num_workers=4, batch_size=self.batch_size, shuffle=False,drop_last=True)#,collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.data_test, num_workers=4, batch_size=self.batch_size, shuffle=False,drop_last=True)#,collate_fn=collate_fn)

class IEMOCAP(LightningDataModule):
    def __init__(self,data_path,  batch_size, num_class,type1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
        self.num_class = num_class
        self.root = data_path
        self.names, self.labels = self.read_names_labels()
        self.train_eval_idxs= self.split_indexes_using_session()
        self.dataset = get_datasets(self.root, self.names, self.labels,type1)
    
    def get_loaders(self):
        train_loaders = []
        eval_loaders = []
        test_loaders =[]
        for ii in range(len(self.train_eval_idxs)):
            train_idxs = self.train_eval_idxs[ii][0]
            eval_idxs  = self.train_eval_idxs[ii][1]
            test_idxs = self.train_eval_idxs[ii][2]
            train_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                      num_workers=self.num_workers,
                                      collate_fn=self.dataset.collater,
                                      drop_last=True)#,
                                      #pin_memory=True)
            eval_loader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     sampler=SubsetRandomSampler(eval_idxs),
                                     num_workers=self.num_workers,
                                     collate_fn=self.dataset.collater,
                                     drop_last=True)#,
                                     #pin_memory=True)
            test_loader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     sampler=SubsetRandomSampler(test_idxs),
                                     num_workers=self.num_workers,
                                     collate_fn=self.dataset.collater,
                                     drop_last=True)#,
                                     #pin_memory=True)
            train_loaders.append(train_loader)
            eval_loaders.append(eval_loader)
            test_loaders.append(test_loader)

        return train_loaders, eval_loaders, test_loaders

    def read_names_labels(self):
        names, labels = [], []
        data = pd.read_csv(self.root + 'IEMOCAP.csv')
        data = data[data['Titel'] != 'Ses03M_script03_2_M003']

        if self.num_class == 4:
            data = data[data['Emotion'].isin(['ang', 'hap', 'sad', 'neu','exc'])]
            emotion_dict = {'ang': 0, 'hap': 1,'exc':1, 'sad': 2, 'neu': 3}
        else:
            data = data[data['Emotion'].isin(['ang', 'hap', 'sad', 'neu', 'exc', 'fru', 'fea', 'sur'])]
            emotion_dict = {'ang': 0, 'hap': 1, 'sad': 2, 'neu': 3, 'exc': 4, 'fru': 5, 'fea': 6, 'sur': 7}
        data['Emotion_cont'] = data['Emotion_cont'].apply(lambda x: x.split('[')[1].split(',')[0])
        data['Emotion_cont'] =data['Emotion_cont'].apply(lambda x: float(x))

        for idx, row in data.iterrows():
            name = row['Titel']
            emotion = emotion_dict[row['Emotion']]
            valence = row['Emotion_cont']
            names.append(name)
            labels.append([emotion, valence])
        
        return names, labels

    ## Split indexes using session for train, validation, and test
    def split_indexes_using_session1(self):
        session_to_idx = {}
        for idx, vid in enumerate(self.names):
            session = int(vid[4]) - 1
            if session not in session_to_idx: session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == 5, f'Must split into five sessions'

        test_idxs = session_to_idx[4]  # Session 5 for testing
        val_idxs = session_to_idx[3]   # Session 1 for validation
        train_idxs = []
        for ii in range(0, 3):         # Sessions 2, 3, 4 for training
            train_idxs.extend(session_to_idx[ii])

        return train_idxs, val_idxs, test_idxs

    def split_indexes_using_session(self):
        # Gain index for cross-validation
        session_to_idx = {}
        for idx, vid in enumerate(self.names):
            session = int(vid[4]) - 1
            if session not in session_to_idx:
                session_to_idx[session] = []
            session_to_idx[session].append(idx)
        assert len(session_to_idx) == 5, 'Must split into five sessions'

        splits = []
        
        for test_session in range(5):
            val_session = (test_session - 1) % 5
            train_sessions = [s for s in range(5) if s != test_session and s != val_session]
            
            test_idxs = session_to_idx[test_session]
            val_idxs = session_to_idx[val_session]
            train_idxs = []
            for session in train_sessions:
                train_idxs.extend(session_to_idx[session])
            
            splits.append([train_idxs, val_idxs, test_idxs])
    
        return splits


class MER2023(LightningDataModule):
    def __init__(self,data_path,  batch_size, num_class,type1):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
        self.num_folder = 5
        self.num_class = num_class
        self.root = data_path
        self.type1 = type1
        data_type = 'train'
        self.names, self.labels = self.read_names_labels(self.root, data_type)
        whole_num = len(self.names)
        self.train_eval_idxs = self.random_split_indexes(whole_num, self.num_folder)
        self.dataset = get_datasets(self.root, self.names, self.labels,self.type1)
     
    def get_loaders(self):
         ## gain train and eval loaders
        train_loaders = []
        eval_loaders = []
        for ii in range(len(self.train_eval_idxs)):
            train_idxs = self.train_eval_idxs[ii][0]
            eval_idxs  = self.train_eval_idxs[ii][1]
            train_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                      num_workers=self.num_workers,
                                      collate_fn=self.dataset.collater,
                                      drop_last=True)#,
                                      #pin_memory=True)
            eval_loader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     sampler=SubsetRandomSampler(eval_idxs),
                                     num_workers=self.num_workers,
                                     collate_fn=self.dataset.collater,
                                     drop_last=True)#,
                                     #pin_memory=True)
            train_loaders.append(train_loader)
            eval_loaders.append(eval_loader)

        test_loaders = []
        for data_type in ['test1', 'test2', 'test3']:
            names, labels = self.read_names_labels(self.root, data_type)
            print (f'{data_type}: sample number {len(names)}')
            test_dataset = get_datasets(self.root, names, labels,self.type1)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     collate_fn=test_dataset.collater,
                                     shuffle=False,
                                     drop_last=True)#,
                                     #pin_memory=True)
            test_loaders.append(test_loader)
        return train_loaders, eval_loaders, test_loaders
    
    # read (names, labels)
    def read_names_labels(self, label_path, data_type, debug=False):
        path = label_path + "label-6way_with_test.npz"
        names, labels = [], []
        emos_mer = ['neutral', 'angry', 'happy', 'sad', 'worried',  'surprise']
        emo2idx_mer, idx2emo_mer = {}, {}
        for ii, emo in enumerate(emos_mer): emo2idx_mer[emo] = ii
        for ii, emo in enumerate(emos_mer): idx2emo_mer[ii] = emo
        assert data_type in ['train', 'test1', 'test2', 'test3']
        if data_type == 'train': corpus = np.load(path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'test1': corpus = np.load(path, allow_pickle=True)['test1_corpus'].tolist()
        if data_type == 'test2': corpus = np.load(path, allow_pickle=True)['test2_corpus'].tolist()
        if data_type == 'test3': corpus = np.load(path, allow_pickle=True)['test3_corpus'].tolist()
        for name in corpus:
            names.append(name)
            labels.append(corpus[name])
        # post process for labels
        for ii, label in enumerate(labels):
            emo = emo2idx_mer[label['emo']]
            if 'val' not in label or label['val'] == '':
                val = -10
            else:
                val = label['val']
            labels[ii] = {'emo': emo, 'val': val}
        # for debug
        if debug: 
            names = names[:100]
            labels = labels[:100]
        return names, labels


    ## 生成 n-folder 交叉验证需要的index信息
    def random_split_indexes(self, whole_num, num_folder):

        # gain indices for cross-validation
        indices = np.arange(whole_num)
        random.shuffle(indices)

        # split indices into five-fold
        whole_folder = []
        each_folder_num = int(whole_num / num_folder)
        for ii in range(num_folder-1):
            each_folder = indices[each_folder_num*ii: each_folder_num*(ii+1)]
            whole_folder.append(each_folder)
        each_folder = indices[each_folder_num*(num_folder-1):]
        whole_folder.append(each_folder)
        assert len(whole_folder) == num_folder
        assert sum([len(each) for each in whole_folder if 1==1]) == whole_num

        ## split into train/eval
        train_eval_idxs = []
        for ii in range(num_folder): # ii in [0, 4]
            eval_idxs = whole_folder[ii]
            train_idxs = []
            for jj in range(num_folder):
                if jj != ii: train_idxs.extend(whole_folder[jj])
            train_eval_idxs.append([train_idxs, eval_idxs])
        
        return train_eval_idxs




class MER4(LightningDataModule):
    def __init__(self,data_path,  batch_size, num_class,type1,no_samples):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = 4
        self.num_folder = 5
        self.num_class = num_class
        self.root = data_path
        self.type1 = type1
        self.no_samples = no_samples
        self.data_type = data_type = 'train'
        self.names, self.labels = self.read_names_labels(self.root, data_type,no_samples)
        whole_num = len(self.names)
        if no_samples < 2000:
            self.train_eval_idxs = self.random_split_indexes1(self.names,self.labels, self.num_folder)
        else:
            self.train_eval_idxs = self.random_split_indexes(whole_num, self.num_folder)
        self.dataset = get_datasets(self.root, self.names, self.labels,self.type1)
     
    def get_loaders(self):
         ## gain train and eval loaders
        train_loaders = []
        eval_loaders = []
        for ii in range(len(self.train_eval_idxs)):
            train_idxs = self.train_eval_idxs[ii][0]
            eval_idxs  = self.train_eval_idxs[ii][1]
            train_loader = DataLoader(self.dataset,
                                      batch_size=self.batch_size,
                                      sampler=SubsetRandomSampler(train_idxs), # random sampler will shuffle index
                                      num_workers=self.num_workers,
                                      collate_fn=self.dataset.collater)#,
                                      #pin_memory=True)
            eval_loader = DataLoader(self.dataset,
                                     batch_size=self.batch_size,
                                     sampler=SubsetRandomSampler(eval_idxs),
                                     num_workers=self.num_workers,
                                     collate_fn=self.dataset.collater)#,
                                     #pin_memory=True)
            train_loaders.append(train_loader)
            eval_loaders.append(eval_loader)

        test_loaders = []
        for data_type in ['test1', 'test2', 'test3']:
            names, labels = self.read_names_labels(self.root, data_type,self.no_samples)
            print (f'{data_type}: sample number {len(names)}')
            test_dataset = get_datasets(self.root, names, labels,self.type1)
            test_loader = DataLoader(test_dataset,
                                     batch_size=self.batch_size,
                                     num_workers=self.num_workers,
                                     collate_fn=test_dataset.collater,
                                     shuffle=False)#,
                                     #pin_memory=True)
            test_loaders.append(test_loader)
        return train_loaders, eval_loaders, test_loaders

    def func_uniform_labels(self, corpus,mapping,emo2idx):
        names, labels = [], []
        for name in corpus:
            label = corpus[name]
            if label['emo'] in mapping:
                label['emo'] = emo2idx[mapping[label['emo']]]
                names.append(name)
                labels.append(label)
        return names, labels
    
    # read (names, labels)
    def read_names_labels(self, label_path, data_type,no_samples, debug=False):
        path = label_path + "label-6way_with_test.npz"
        emo2idx = {'happy': 1, 'sad': 2, 'neutral': 3, 'angry': 0}
        mapping = {'neutral': 'neutral', 'angry': 'angry', 'happy': 'happy', 'sad': 'sad'}
        assert data_type in ['train', 'test1', 'test2', 'test3']
        if data_type == 'train': corpus = np.load(path, allow_pickle=True)['train_corpus'].tolist()
        if data_type == 'test1': corpus = np.load(path, allow_pickle=True)['test1_corpus'].tolist()
        if data_type == 'test2': corpus = np.load(path, allow_pickle=True)['test2_corpus'].tolist()
        if data_type == 'test3': corpus = np.load(path, allow_pickle=True)['test3_corpus'].tolist()
        names, labels = self.func_uniform_labels(corpus,mapping,emo2idx)
        if data_type == 'train':
            # Dictionary to hold the indices of each class
            class_indices = defaultdict(list)
            
            # Fill the dictionary with indices for each class
            for idx, label in enumerate(labels):
                class_indices[label['emo']].append(idx)
            
            # List to hold the final selected indices
            selected_indices = []
            
            # For each class, randomly sample the required number of indices
            for label, indices in class_indices.items():
                if len(indices) > no_samples:
                    selected_indices.extend(random.sample(indices, no_samples))
                else:
                    selected_indices.extend(indices)  # If less samples than required, take all
            
            # Filter names and labels based on the selected indices
            names = [names[idx] for idx in selected_indices]
            labels = [labels[idx] for idx in selected_indices]
        return names, labels

    def random_split_indexes1(self, names, labels, num_folder):
        class_indices = defaultdict(list)
        
        # Group indices by class
        for idx, label in enumerate(labels):
            class_indices[label['emo']].append(idx)
        
        # Ensure each class is equally represented in each fold
        folds = [[] for _ in range(num_folder)]
        
        for indices in class_indices.values():
            random.shuffle(indices)
            fold_sizes = [len(indices) // num_folder] * num_folder
            for i in range(len(indices) % num_folder):
                fold_sizes[i] += 1
            
            current_idx = 0
            for fold, size in zip(folds, fold_sizes):
                fold.extend(indices[current_idx:current_idx + size])
                current_idx += size
        
        train_eval_idxs = []
        for i in range(num_folder):
            eval_idxs = folds[i]
            train_idxs = [idx for fold in folds if fold != eval_idxs for idx in fold]
            train_eval_idxs.append([train_idxs, eval_idxs])
        
        return train_eval_idxs


    ## 生成 n-folder 交叉验证需要的index信息
    def random_split_indexes(self, whole_num, num_folder):
        if whole_num < num_folder:
            # Create folds equal to the number of samples if fewer samples than folds
            indices = np.arange(whole_num)
            random.shuffle(indices)
            
            train_eval_idxs = []
            for i in range(whole_num):
                eval_idxs = [indices[i]]
                train_idxs = [indices[j] for j in range(whole_num) if j != i]
                train_eval_idxs.append([train_idxs, eval_idxs])
            
            return train_eval_idxs
        else:
            # Gain indices for cross-validation
            indices = np.arange(whole_num)
            random.shuffle(indices)

            # Split indices into k-fold
            whole_folder = []
            each_folder_num = int(whole_num / num_folder)
            for ii in range(num_folder - 1):
                each_folder = indices[each_folder_num * ii: each_folder_num * (ii + 1)]
                whole_folder.append(each_folder)
            each_folder = indices[each_folder_num * (num_folder - 1):]
            whole_folder.append(each_folder)

            assert len(whole_folder) == num_folder
            assert sum([len(each) for each in whole_folder if 1 == 1]) == whole_num

            # Split into train/eval
            train_eval_idxs = []
            for ii in range(num_folder):  # ii in [0, 4]
                eval_idxs = whole_folder[ii]
                train_idxs = []
                for jj in range(num_folder):
                    if jj != ii:
                        train_idxs.extend(whole_folder[jj])
                train_eval_idxs.append([train_idxs, eval_idxs])

            return train_eval_idxs