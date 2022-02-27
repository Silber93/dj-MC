import os
import random
import numpy as np
import torch
from utils import calc_dist, delta_medoids
from user_interaction import choose_rep_songs, show_song
from custom_datasets import ProcessedSongDataset
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
import matplotlib.pyplot as plt
import os
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = 'plots'

class DJMC:
    def __init__(self,
                 song_dataset: ProcessedSongDataset,
                 saved_data: dict,
                 bins: int = 10,
                 n_descriptors: int = 34,
                 dist_metric: str = 'LNorm_2',
                 play_dur: int = 10,
                 raiting_dur: int = 5,
                 upper_m_portion=2):

        self.bins = bins
        self.n_descriptors = n_descriptors
        self.dist_metric = dist_metric
        self.upper_m_portion = upper_m_portion
        self.song_dataset = song_dataset
        self.play_dur = play_dur
        self.rating_dur = raiting_dur
        self.upper_m, self.upper_m_vecs = [], []

        self.k_s, self.phi_s, self.k_t, self.phi_t = None, None, None, None
        if saved_data is not None:
            self.k_s, self.phi_s = saved_data['k_s'], saved_data['phi_s']
            self.k_t, self.phi_t = saved_data['k_t'], saved_data['phi_t']

    def save_user_data(self, user_dir: str):
        saved_data = {'k_s': self.k_s, 'phi_s': self.phi_s,
                      'k_t': self.k_t, 'phi_t': self.phi_t}
        torch.save(saved_data, f'{user_dir}/saved_data.pkl')

    def transition_vec(self, i: int, j: int) -> torch.Tensor:
        a_i = self.song_dataset[i][1]
        a_j = self.song_dataset[j][1]
        one_hots = []
        for i in range(self.n_descriptors):
            a_i_i = a_i[i*self.bins:(i+1)*self.bins].view(-1, 1).float()
            a_j_i = a_j[i*self.bins:(i+1)*self.bins].view(1, -1).float()
            one_hots.append(torch.matmul(a_i_i, a_j_i).view(-1).int())
        ret = torch.cat(one_hots)
        return ret.float()

    def calc_rs(self, i):
        a_i = self.song_dataset[i][1]
        return torch.dot(self.phi_s, a_i.float())

    def init_phi_s(self, preferred_set):
        k_s = len(preferred_set)
        chosen_songs = [self.song_dataset[i][1] for i in preferred_set]
        vec_shape = self.bins * self.n_descriptors
        phi_s = (torch.ones(vec_shape) / (k_s + self.bins)).to(DEVICE)
        for i in range(k_s):
            phi_s += (1/(k_s + 1)) * chosen_songs[i]
        return k_s, phi_s

    def update_upper_m(self):
        rs_m = [(idx, self.calc_rs(idx)) for idx in self.song_dataset.true_idx]
        rs_m = sorted(rs_m, reverse=True, key=lambda x: x[1])[:int(len(self.song_dataset.true_idx) * self.upper_m_portion)]
        rs_m = [x[0] for x in rs_m]
        self.upper_m = rs_m
        self.upper_m_vecs = [[x.item() for x in self.song_dataset[i][1]] for i in self.upper_m]
    # def get_delta(self):
    #     return torch.Tensor([56])
    #     all_distances = []
    #     for i in tqdm(range(len(self.song_dataset))):
    #         for j in range(i+1, len(self.song_dataset)):
    #             all_distances.append(calc_dist(i, j, self.song_dataset, self.dist_metric))
    #     q = torch.Tensor([0.1])
    #     all_distances = torch.Tensor(all_distances)
    #     delta = torch.quantile(all_distances, q)
    #     return delta.item()

    def init_phi_t(self, preferred_set):
        vec_shape = self.n_descriptors * (self.bins**2)
        upper_m = self.get_upper_m()
        # delta = self.get_delta().to(DEVICE)
        # rep_set = delta_medoids(upper_m, self.song_dataset, delta, self.dist_metric)
        upper_m_vecs = [[x.item() for x in self.song_dataset[i][1]] for i in upper_m]
        n_clusters = min(100, int(len(upper_m) / 20))
        upper_m_dict = {i: upper_m[i] for i in range(len(upper_m))}
        kmedoids = KMedoids(n_clusters=n_clusters, random_state=0)
        kmedoids.labels_ = upper_m
        kmedoids = kmedoids.fit(upper_m_vecs)
        kmedoids_rep_set = kmedoids.medoid_indices_
        rep_set = [upper_m_dict[x] for x in kmedoids_rep_set]
        rep_set = [x for x in rep_set if x not in preferred_set]
        rep_songs = [int(np.random.choice(rep_set, 1))]
        rep_songs += choose_rep_songs(rep_set, self.song_dataset, self.play_dur)
        k_t = len(rep_songs)
        phi_t = torch.ones(vec_shape).to(DEVICE) / (k_t + self.bins)
        for i in range(1, k_t):
            phi_t += (1/(k_t+1)) * self.transition_vec(rep_songs[i-1], rep_songs[i])
        return k_t, phi_t, upper_m

    def init_parmas(self, preferred_set):
        self.k_s, self.phi_s = self.init_phi_s(preferred_set)
        self.k_t, self.phi_t, self.upper_m = self.init_phi_t(preferred_set)

    def model_update(self):
        print()
        rewards = []
        selected_songs = []
        offered_songs = []
        all_payoffs = []
        print(f"We are about to find {self.rating_dur} songs for you to rate.\nPlease wait until we find the first song for you")
        for i in range(self.rating_dur):
            selected_song_idx, offered_songs, payoffs = self.plan_via_tree_search(self.play_dur,
                                                                                  i,
                                                                                  self.rating_dur,
                                                                                  offered_songs=offered_songs)
            all_payoffs.append(payoffs)
            selected_songs.append(selected_song_idx)
            # plan_hor -= 1 # eventually the tree search returns only one song so there is no need for decreasing the
            # planning horizon length
            print("chosen song for you:")
            show_song(selected_song_idx, self.song_dataset)
            reward = int(input("How much, on a 1 to 5 scale, did you enjoy this song choice? Please type a number: "))
            if i < self.rating_dur-1:
                print("\nSearching for the next song in your playlist")
            else:
                print("\nThank you! Please wait until we find a playlist that suits your preferences...")
            rewards.append(reward)
            if i == 0:
                continue
            rewards_avg = sum(rewards[:-1]) / len(rewards[:-1])
            r_incr = np.log(reward / rewards_avg)
            ### weight update ###
            rs_ai = self.calc_rs(selected_song_idx)
            theta_t = self.transition_vec(selected_songs[-1], selected_songs[-2])
            theta_s = self.song_dataset[selected_song_idx][1].float()
            r_t = torch.dot(self.phi_t, theta_t)
            w_s = rs_ai / (rs_ai + r_t)
            w_t = r_t / (rs_ai + r_t)
            self.phi_s = i / (i + 1) * self.phi_s + i / (i + 1) * theta_s * w_s * r_incr
            self.phi_t = i / (i + 1) * self.phi_t + i / (i + 1) * theta_t * w_t * r_incr
            ## normalize per descriptor ##
            self.phi_s /= self.phi_s.shape[0]
            self.phi_t /= self.phi_t.shape[0]
        return offered_songs, all_payoffs

    def plan_via_tree_search(self,
                             plan_horizon: int,
                             i: int,
                             tot: int,
                             epochs=100,
                             use_songs_type=False,
                             n_clusters=10,
                             offered_songs=[]):

        """ Returns the best song idx"""
        upper_m_vecs = self.upper_m_vecs
        if use_songs_type:
            kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(upper_m_vecs)
        best_traj = None
        highest_exp_payoff = -np.infty
        payoffs = []
        for _ in tqdm(range(epochs), desc=f'song {i+1}/{tot}'):
            traj = []
            for i in range(plan_horizon):
                if use_songs_type:
                    cluster = random.choice(range(n_clusters))
                    labels = list(kmeans.labels_)
                    cluster_indices = [index for index, element in enumerate(labels) if element == cluster]
                    song = random.choice(cluster_indices)
                    # song = kmeans.cluster_centers_[cluster]  # taking the centroid of the chosen cluster as an
                    # abstract song representative
                else:
                    song = random.choice(range(len(upper_m_vecs)))
                traj.append(song)
            exp_traj_payoff = self.calc_rs(traj[0])
            for i in range(1, plan_horizon):
                curr_traj = traj[:i + 1]
                exp_traj_payoff += self.calc_sequence_reward(curr_traj)
                exp_traj_payoff += self.calc_rs(traj[i])
            if exp_traj_payoff > highest_exp_payoff:
                highest_exp_payoff = exp_traj_payoff
                best_traj = traj
            payoffs.append(highest_exp_payoff.cpu())
        offered_songs.append(best_traj[0])
        return best_traj[0], offered_songs, payoffs

    def calc_sequence_reward(self, sequence: list):
        seq_len = len(sequence)
        reward = 0
        for i in range(1, seq_len):
            theta_t = self.transition_vec(sequence[i-1], sequence[i])
            reward += torch.dot(self.phi_t, theta_t) / (i ** 2)
        return reward








