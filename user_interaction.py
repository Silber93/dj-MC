import random
import os
import torch
from custom_datasets import ProcessedSongDataset


def load_user(main_user_dir):
    print(f'welcome to the Million Song Dataset!')
    print("enter your username. if you wish to create a new username or reset an existing username, "
          "type the sign ! at the beginning of the username.")
    while True:
        username = input("enter username:")
        new_user = False
        if username[0] == '!':
            new_user = True
            username = username[1:]
        user_path = f'{main_user_dir}/{username}'
        is_existing_user = os.path.exists(user_path)
        if new_user:
            if not is_existing_user:
                os.mkdir(user_path)
            return user_path, None
        if not is_existing_user:
            response = ''
            while response not in ['y', 'n']:
                response = input('no username found, do you want to create this username? [y/n]')
            if response == 'y':
                os.mkdir(user_path)
                return user_path, None
            print("please enter a different username")
            continue
        saved_data = torch.load(f'{user_path}/saved_data.pkl')
        return user_path, saved_data


def saved_data_actions():
    print("You can browse your saved playlist by typing the char b or create a new playlist by typing the char p"
          " (Any other input will terminate the program)")
    res = input("[b/p]:")
    return res


def choose_initial_preferences(song_dataset: ProcessedSongDataset, init_song_ids):
    n = len(song_dataset)
    print("choose song ids out of this list:")
    if type(init_song_ids) == int:
        i = 0
        sampled_songs = []
        while i < init_song_ids:
            sampled_idx = random.sample(song_dataset.true_idx, 1)[0]
            if sampled_idx not in sampled_songs:
                i += 1
        for idx in sorted(sampled_songs):
            show_song(idx, song_dataset)

    else:
        for idx in init_song_ids:
            show_song(idx, song_dataset)
    chosen_songs = input("\nNotice: put the cursor right after the colon"
                         "\nchoose songs by IDs (separated by single space):").split()
    chosen_songs = [int(idx) for idx in chosen_songs]
    print("\nyour songs are:")
    for idx in chosen_songs:
        show_song(idx, song_dataset)
    print("\nSearching for the best songs for you...")
    return chosen_songs


def show_song(idx, song_dataset):
    idx = int(idx)
    title = song_dataset[idx][0]["title"]
    artist = song_dataset[idx][0]["artist_name"]
    album = song_dataset[idx][0]["album"]
    print(f'ID {idx}: \"{title}\" by {artist} ({album})')


def choose_rep_songs(song_ids, song_dataset, play_dur):
    print(f"Please choose a subset of songs you would like out of these {len(song_ids)} songs")
    for idx in sorted(song_ids):
        show_song(idx, song_dataset)
    chosen_songs = input("choose songs by IDs (separated by single space): ").split()
    print("\nYou choosed the next songs:")
    for idx in chosen_songs:
        show_song(idx, song_dataset)
    return [int(x) for x in chosen_songs]
