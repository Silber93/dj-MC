import os
import torch
import numpy as np
from custom_datasets import RawSongDataset, ProcessedSongDataset
from utils import calc_dist
from tqdm import tqdm
import datetime as dt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def presave_dist_matrix(path: str, perc: float, dist_metric='sub_abs'):
    rebuilt_song_pkl = f'{path}/rebuilt_dataset_{perc}.pkl'
    song_dataset = ProcessedSongDataset(rebuilt_song_pkl)
    n = len(song_dataset)
    total = int(((n-1) * n) / 2)
    p_bar = tqdm(total=total)
    for i in range(n):
        for j in range(i+1, n):
            d = calc_dist(i, j, song_dataset, dist_metric)
            p_bar.update()
    p_bar.close()
    save_path = f'{path}/dist_matrix_{dist_metric}_{perc}.pkl'
    torch.save(song_dataset.dist_matrix, save_path)


def basic_preproccess_save(v1_dataset: RawSongDataset, v1_dir, v15_path):
    all_data_old, all_desc_old = [], []
    if os.path.exists(v15_path):
        old_save = torch.load(v15_path)
        all_data_old = old_save['all_data']
        all_desc_old = old_save['all_desc']
    saved_tracks = []
    if len(all_desc_old) > 0:
        for desc in all_desc_old:
            track_id = desc['track_id']
            track_path = f'{v1_dir}/{track_id[2]}/{track_id[3]}/{track_id[4]}/{track_id}.h5'
            saved_tracks.append(track_path)
        print(f"found {len(saved_tracks)} saved tracks")
    new_song_locations = []
    t = dt.datetime.now().replace(microsecond=0)
    for x in tqdm(v1_dataset.song_location_dataset, desc=f'{t} - creating new song list'):
        if x not in saved_tracks:
            new_song_locations.append(x)
    v1_dataset.song_location_dataset = new_song_locations
    new_data, new_desc = load_v1_data(v1_dataset)
    all_data = all_data_old + new_data
    all_desc = all_desc_old + new_desc
    save_pkl = {'all_data': all_data, 'all_desc': all_desc}
    torch.save(save_pkl, v15_path)


def create_onehot_and_save(all_data: list,
                           all_desc: list,
                           track2idx: dict,
                           global_percentiles,
                           new_dataset_pkl_path: str):
    message = f'{dt.datetime.now().replace(microsecond=0)} - CREATING INDEX DATA'
    data_dict = {}
    n = len(all_data)
    all_data = torch.Tensor(all_data)
    for i in tqdm(range(n), desc=message):
        vec, description = all_data[i].view(-1, 1), all_desc[i]
        new_vec = []
        for j, x in enumerate(vec):
            p = torch.Tensor(global_percentiles[:, j])
            idx = (x >= p).sum() - 1
            new_vec.append(idx.item())
        data_dict[i] = {'data': torch.LongTensor(new_vec), 'desc': description}
    save_dict = {'data_dict': data_dict, 'track2idx': track2idx}
    torch.save(save_dict, new_dataset_pkl_path)


def collect_percentile_data(all_data: list):
    torched_data = np.array(all_data)
    q = np.array([0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    percentiles = np.quantile(torched_data, q, axis=0)
    return percentiles


def load_v1_data(v1_dataset: RawSongDataset) -> [list, list]:
    load_message = f'{dt.datetime.now().replace(microsecond=0)} - LOADING DATA'
    all_data, all_desc = [], []
    songs_dumped = 0
    n = len(v1_dataset)
    p_bar = tqdm(total=n, desc=load_message)
    for i in range(n):
        try:
            desc, vec = v1_dataset[i]
            all_data.append(vec)
            all_desc.append(desc)
        except:
            songs_dumped += 1
            p_bar.desc = f"{load_message} (songs dumped: {songs_dumped})"
        p_bar.update(1)
    p_bar.close()
    return all_data, all_desc


def rebuild_dataset(v1_dir: str,
                    v2_dir: str,
                    data_features: list,
                    desc_features: list,
                    type_features: list,
                    sample_proportion: float = 0.01,
                    v15_path=None):
    if not os.path.exists(v2_dir):
        os.mkdir(v2_dir)
    # percentile_path = f'{v2_dir}/glob_percentiles_{sample_proportion}.pkl'
    v1_dataset = RawSongDataset(v1_dir, data_features, desc_features, type_features)
    num_of_songs = int(len(v1_dataset) * sample_proportion)
    v1_dataset.song_location_dataset = v1_dataset.song_location_dataset[:num_of_songs]
    new_dataset_pkl_path = f'{v2_dir}/rebuilt_dataset_{sample_proportion}.pkl'
    if not os.path.exists(new_dataset_pkl_path):
        if v15_path:
            basic_preproccess_save(v1_dataset, v1_dir, v15_path)
            saved_data = torch.load(v15_path)
            all_data, all_desc = saved_data['all_data'], saved_data['all_desc']
        else:
            all_data, all_desc = load_v1_data(v1_dataset)
        track2idx = v1_dataset.track2indx
        global_percentiles = collect_percentile_data(all_data)
        create_onehot_and_save(all_data, all_desc, track2idx, global_percentiles, new_dataset_pkl_path)



DATA_FEATURES = ['beats_start', 'segments_loudness_max', 'segments_pitches', 'segments_timbre']
SONG_DESC = [('metadata', 'songs'), {9: 'artist_name', 14: 'album', 18: 'title', 17: 'song_id'}]
SONG_TYPE_FEATURES = ['artist_terms', 'artist_terms_freq']
RAW_DIR = 'data/MillionSongSubset' if DEVICE == 'cpu' else 'data'
PROCESSED_DIR = 'rebuilt_data'
V15_PATH = 'rebuilt_data/middle_state.pkl'


if __name__ == '__main__':
    percs = [1]
    print(percs)
    print(f"running preprocess ({DEVICE})")
    for perc in percs:
        perc = round(perc, 2)
        print(perc)
        rebuild_dataset(RAW_DIR, PROCESSED_DIR, DATA_FEATURES, SONG_DESC, SONG_TYPE_FEATURES,
                        sample_proportion=perc, v15_path=V15_PATH)
        # preprocess.presave_dist_matrix(processed_dir, perc)




