import os
import torch
from custom_datasets import ProcessedSongDataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def delete_saved_data(user_dir):
    saved_playlists = [f'{user_dir}/{x}' for x in os.listdir(user_dir) if 'playlist' in x]
    for playlist in saved_playlists:
        os.remove(playlist)


def save_user_playlist(playlist, user_dir):
    saved_playlists = [x for x in os.listdir(user_dir) if 'playlist' in x]
    if len(saved_playlists) == 0:
        mx = 1
    else:
        mx = max([int(x.split('_')[-1].replace('.pkl', '')) for x in saved_playlists]) + 1
    torch.save({'idx': playlist}, f'{user_dir}/playlist_{mx}.pkl')


def get_user_playlist(user_dir):
    playlists = []
    for x in os.listdir(user_dir):
        if 'playlist' in x:
            playlists.append(torch.load(f'{user_dir}/{x}')['idx'])
    return playlists


def plot_and_save(data, title, save_dir):
    plt.figure()
    plt.plot(range(len(data)), data)
    plt.xlabel('epoch')
    plt.ylabel('payoff')
    plt.title(f'average payoffs - {title}')
    title = title if title == 'init' else 'auto'
    plt.savefig(f'{save_dir}/payoffs_{title}.png')


def calc_avgs(data):
    avgs = []
    cols = len(data[0])
    for j in range(cols):
        avgs.append(sum([data[i][j] for i in range(len(data))]) / cols)
    return avgs


def calc_dist(i, j, song_dataset, dist_metric):
    i, j = min(i, j), max(i, j)
    if (i, j) in song_dataset.dist_matrix:
        return song_dataset.dist_matrix[(i, j)]
    a_i, a_j = song_dataset[i][1], song_dataset[j][1]
    if 'LNorm' in dist_metric:
        p = float(dist_metric.split('_')[1])
        d = torch.cdist(a_i.view(1, -1).float(), a_j.view(1, -1).float(), p=p).to(DEVICE)
    if dist_metric == 'sub_abs':
        d = torch.sum(torch.abs(a_i-a_j))
    song_dataset.dist_matrix[(i, j)] = d.item()
    return d


def RepAssign(clusters, songs, song_dataset, delta, clusters_reps, dist_metric):
    for i in range(len(songs)):
        dist = np.infty
        rep = None
        for j, _rep in enumerate(clusters_reps):
            d = calc_dist(songs[i], _rep, song_dataset, dist_metric)
            # TODO: equal???
            if j == 0 or d < dist:
                rep, dist = _rep, d
        if dist <= delta:
            if rep in clusters_reps:
                for _rep in clusters_reps:
                    if songs[i] in clusters[_rep]:
                        clusters[_rep].remove(songs[i])
                clusters[rep].add(songs[i])
            else:
                clusters[rep] = set(songs[i])
        else:
            rep = songs[i]
            clusters[rep] = {songs[i]}
            clusters_reps.append(songs[i])
    return clusters


def delta_medoids(songs: list,
                  song_dataset: ProcessedSongDataset,
                  delta: float,
                  dist_metric: str):
    print(f'upper m contains {len(songs)} songs out of {len(song_dataset)} total')
    current_reps = []
    rep_history = [[]]
    clusters = {}
    i = 0
    pbar = tqdm(desc=f"C size: {len(rep_history[-1])}")
    while True:
        print('0')
        clusters = RepAssign(clusters, songs, song_dataset, delta, current_reps, dist_metric)
        print('1')
        rep_history.append([])
        new_clusters = {}
        for c in clusters:
            print(c, len(clusters[c]))
            rep_args = {}
            for song_a in tqdm(clusters[c]):
                tot_dist = 0
                for song_b in clusters[c]:
                    if song_a == song_b:
                        continue
                    d = calc_dist(song_a, song_b, song_dataset, dist_metric)
                    if d <= delta:
                        tot_dist += d
                rep_args[song_a] = tot_dist
            # rep_args = {song: sum([calc_dist(x, song, song_dataset, dist_metric)
            #                        for x in clusters[c] if calc_dist(x, song, song_dataset, dist_metric) <= delta])
            #             for song in clusters[c]}
            # print(c, rep_args)
            new_reps = min(rep_args, key=rep_args.get)
            rep_history[-1].append(new_reps)
            new_clusters[new_reps] = clusters[c]
        print('2')
        current_reps = rep_history[-1]
        rep_history[-1] = sorted(rep_history[-1])
        change = len([x for x in rep_history[-1] if x not in rep_history[-2]]) + \
                 len([x for x in rep_history[-2] if x not in rep_history[-1]])
        # print(i, rep_history[-1], len(rep_history[-1]), change)
        # TODO: think about stopping condition
        print('3')
        if rep_history[-1] in rep_history[:-1]:
            break
        if i >= len(song_dataset) and len(rep_history[-1]) == len(rep_history[-100]):
            break
        clusters = new_clusters
        i += 1
        print('4')
        pbar.desc = f"C size: {len(rep_history[-1])}"
        pbar.update(1)
        print('5')
    pbar.close()
    return rep_history[-1]
