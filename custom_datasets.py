import os
import h5py
from torch.utils.data import Dataset
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import random
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


class RawSongDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 features: list,
                 description: list,
                 song_type: list):

        self.data_path = data_path
        self.data_features = features
        self.desc_features = description
        self.type_features = song_type
        self.song_location_dataset = self.create_location_dataset(data_path)
        self.track2indx = {}

    def create_location_dataset(self, data_path):
        song_location_dataset = []
        for file in sorted(os.listdir(data_path)):
            file_path = f'{data_path}/{file}'
            if os.path.isdir(file_path):
                song_location_dataset += self.create_location_dataset(file_path)
            else:
                if 'DS_Store' in file_path:
                    continue
                song_location_dataset.append(file_path)
        return song_location_dataset

    def get_relevant_features(self, h5):
        features, description = {}, {}
        artist_terms, artist_terms_freq = [], []
        for k in h5:
            for subkey in h5[k]:
                if subkey in self.data_features:
                    features[subkey] = list(h5[k][subkey])
                if k == self.desc_features[0][0] and subkey == self.desc_features[0][1]:
                    metadata = h5[k][subkey][0]
                    desc_idx = self.desc_features[1]
                    description = {desc_idx[idx]: metadata[idx].decode('UTF-8') for idx in desc_idx}
                if subkey == self.type_features[0]:
                    artist_terms = list(h5[k][subkey])
                if subkey == self.type_features[1]:
                    artist_terms_freq = list(h5[k][subkey])
        best_song_type = sorted(enumerate(artist_terms_freq), reverse=True, key=lambda x: x[1])[0][0]
        description['song_type'] = artist_terms[best_song_type].decode('UTF-8')
        return description, features

    @staticmethod
    def get_tempo_or_loudness_features(feature_val, is_beats):
        np_val = np.array(feature_val)
        if is_beats:
            prev_vals = [0] + feature_val[:-1]
            np_val = np_val - np.array(prev_vals)
        q = np.array([0.1, 0.9])
        quantiles = np.quantile(np_val, q)
        avg = np_val.mean()
        var = np_val.var()
        refined_features = list(quantiles) + [avg] + [var]
        return refined_features

    @staticmethod
    def get_pitch_or_timbre_features(feature_val):
        feature_val = np.array(feature_val)
        avg_vec = feature_val.mean(axis=0)
        var = avg_vec.var()
        refined_features = list(avg_vec) + [var]
        return refined_features

    def __len__(self):
        return len(self.song_location_dataset)

    def __getitem__(self, idx):
        h5 = h5py.File(self.song_location_dataset[idx])
        description, features = self.get_relevant_features(h5)
        description['track_id'] = self.song_location_dataset[idx].split('/')[-1].replace('.h5', '')
        self.track2indx[description['track_id']] = idx
        tempo_features = self.get_tempo_or_loudness_features(features['beats_start'], True)
        loudness_features = self.get_tempo_or_loudness_features(features['segments_loudness_max'], False)
        pitch_features = self.get_pitch_or_timbre_features(features['segments_pitches'])
        timbre_features = self.get_pitch_or_timbre_features(features['segments_timbre'])
        v1_vector = tempo_features + loudness_features + pitch_features + timbre_features
        return description, v1_vector


class ProcessedSongDataset(Dataset):
    def __init__(self,
                 data_path: str,
                 train_perc,
                 alt_perc=0.1,
                 dist_matrix: str = None):
        print("loading dataset...")
        all_data_dict = torch.load(data_path)
        self.train_perc, self.alt_perc = train_perc, alt_perc
        self.dataset_dict = all_data_dict['data_dict']
        self.track2idx = all_data_dict['track2idx']
        self.dist_matrix = {}
        if dist_matrix is not None:
            self.dist_matrix = torch.load(dist_matrix)
        self.n = len(self.dataset_dict)
        train_size = int(self.n*train_perc)
        self.true_idx = range(train_size)

    def shuffle_data(self):
        shuffle_data_size = int(self.n*self.alt_perc)
        self.true_idx = sorted(random.sample(range(self.n), shuffle_data_size))

    def __len__(self):
        return len(self.dataset_dict)

    def __getitem__(self, idx):
        t_idx = self.true_idx[idx]
        desc = self.dataset_dict[t_idx]["desc"]
        idx_data = self.dataset_dict[t_idx]["data"]
        one_hots = []
        for idx in idx_data:
            one_hots.append(F.one_hot(idx.long(), num_classes=10))
        one_hots = torch.cat(one_hots).to(DEVICE)
        return desc, one_hots
