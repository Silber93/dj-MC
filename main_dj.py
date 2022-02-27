from utils import *
from user_interaction import *
from custom_datasets import ProcessedSongDataset
from custom_models import DJMC
import torch


def adjust_dj_for_user(dj: DJMC, user_dir):
    delete_saved_data(user_dir)
    preferred_set = choose_initial_preferences(dj.song_dataset, 150)
    dj.init_parmas(preferred_set)
    chosen_songs, all_payoffs = dj.model_update()
    all_payoffs = calc_avgs(all_payoffs)
    dj.save_user_data(user_dir)
    plot_and_save(all_payoffs, 'init', user_dir)
    return dj


def create_auto_playlist(dj, user_dir):
    playlist_ids = []
    tracks_ids = []
    all_payoffs = []
    chosen_songs = []
    for i in range(PLAYLIST_DUR):
        selected_song_id, chosen_songs, payoff = dj.plan_via_tree_search(plan_horizon=10,
                                                                         i=i,
                                                                         tot=PLAYLIST_DUR,
                                                                         offered_songs=chosen_songs)
        all_payoffs.append(payoff)
        playlist_ids.append(selected_song_id)
        tracks_ids.append(song_dataset[selected_song_id][0]['track_id'])
    dj.save_user_data(user_dir)
    all_payoffs = calc_avgs(all_payoffs)
    plot_and_save(all_payoffs, 'playlist', user_dir)
    print("This is the playlist we built for you:\n")
    for idx in playlist_ids:
        show_song(idx, song_dataset)
    save_user_playlist(playlist_ids, user_dir)


def browsing_loop(dj, user_dir):
    saved_playlists = get_user_playlist(user_dir)
    np = len(saved_playlists)
    s = '' if np == 1 else 's'
    print(f"\nyou have {np} playlist{s} saved")
    res = saved_data_actions()
    while res in ['b', 'p']:
        if res == 'b':
            for i, playlist in enumerate(saved_playlists):
                print(f'playlist {i+1}:')
                for idx in playlist:
                    show_song(idx, dj.song_dataset)
                print()
        if res == 'p':
            dj.song_dataset.shuffle_data
            dj.update_upper_m()
            create_auto_playlist(dj, user_dir)
            saved_playlists = get_user_playlist(user_dir)
            np = len(saved_playlists)
            s = '' if np == 1 else 's'
            print(f"\nyou have {np} playlist{s} saved")
        res = saved_data_actions()

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
PLAYLIST_DUR = 10
processed_dir = 'rebuilt_data'
user_main_dir = 'users'
train_perc = 0.2
data_path = f'{processed_dir}/rebuilt_dataset.pkl'


song_dataset = ProcessedSongDataset(data_path, train_perc)
user_dir, user_data = load_user(user_main_dir)
dj = DJMC(song_dataset, user_data, dist_metric='sub_abs', upper_m_portion=0.2, raiting_dur=5)

if user_data is None:
    dj = adjust_dj_for_user(dj, user_dir)
    create_auto_playlist(dj, user_dir)

browsing_loop(dj, user_dir)








