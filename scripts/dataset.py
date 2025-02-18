import pandas as pd
import torch
import numpy as np
import datetime
from tqdm import tqdm
from datetime import datetime, timedelta
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Any, Tuple, List, Iterable
from collections import Counter
from numpy.lib.stride_tricks import as_strided
from config import DataConfig
from pathlib import Path


def preprocess():
    current_dir = Path(__file__).parent
    data_path = current_dir.parent/'foursquare_tky'/'foursquare_tky.dyna'
    path = data_path.resolve()

    dyna = pd.read_csv(path)
    dyna.drop(columns=['dyna_id', 'type'], inplace=True)
    return dyna


def sample_dataset(data: pd.DataFrame, top_locs: int, heldout: float = None, min_datapoints: int = 21):
    """
    input:
        data: foursquare_tky
        min_datapoints: choose the user if his checkinpoints more than min_datapoints
        heldout: the ratio of train and test dataset
    output:
        user_vocab: {user: idx}
        dataset: {idx: indexed user's dataframe}. the df contains ['location', 'time', 'week of day']
    ps:
        the user_vocab is used to help __getitem__ in checkin dataset
    """
    hot_points = data['location'].value_counts().keys()[:top_locs].to_list()
    hot_users = data.loc[data['location'].isin(hot_points), 'entity_id'].unique()
    print(f'There are in total {len(hot_users)} users which have been to the top {top_locs} hot points.')

    data_hotpoints = data[data['entity_id'].isin(hot_users)]
    count_user = dict(Counter(data_hotpoints['entity_id'].to_list()))
    users = [user for user, count in count_user.items() if count >= min_datapoints]
    print(f'\n There are in total {len(users)} hotpoints users which have more than {min_datapoints} checkins')

    users_vocab = {
        user: idx for user, idx in
        zip(data_hotpoints['entity_id'].unique(), range(1, len(data_hotpoints['entity_id'].unique()) + 1))
    }
    loc_vocab = {
        loc: idx for loc, idx in
        zip(data_hotpoints['location'].unique(), range(1, len(data_hotpoints['location'].unique()) + 1))
    }

    sampled_data = data[data['entity_id'].isin(users)].copy()
    sampled_data['time'] = sampled_data['time'].apply(lambda t: datetime.strptime(t, '%Y-%m-%dT%H:%M:%SZ'))
    # the n-th hour during a week
    sampled_data['timeslot'] = sampled_data['time'].apply(lambda t: t.weekday() * 24 + t.hour)
    sampled_data['time'] = sampled_data['time'].apply(lambda x: (x - datetime(1970, 1, 1)).total_seconds())

    time_vocab = {
        timeslot: idx for timeslot, idx in
        zip(sampled_data['timeslot'].unique(), range(1, len(sampled_data['timeslot'].unique()) + 1))
    }

    dataset = {}
    for i, user in tqdm(enumerate(users), leave=False, desc='creating sample data', colour='blue'):
        user_data = sampled_data.loc[sampled_data['entity_id'] == user]
        user_data = {
            'location': user_data['location'].to_list(),
            'time': user_data['time'].to_list(),
            'timeslot': user_data['timeslot'].to_list()
        }
        dataset[users_vocab[user]] = user_data

    if heldout is None:
        return dataset, users_vocab, max(list(loc_vocab.keys())), max(list(time_vocab.keys()))
    else:
        tlen = int(heldout * len(dataset))
        train = dict(list(dataset.items())[: tlen])
        valid = dict(list(dataset.items())[tlen:])
        return (train, valid), users_vocab


class CheckinDataset(Dataset):
    def __init__(self, data: dict, traj_len: int, target_len):
        """
        Params:
        - data: 上一步得到的数据集
        - traj_len: 假设观测轨迹的长度是固定的，如果模型设定是根据长度为20的过去数据来进行下一个点的预测，则traj_len=21
        - target_len: 需要预测的轨迹长度
        """
        super().__init__()

        uids = []
        locations = []
        time = []
        timeslots = []

        for user, user_data in tqdm(data.items()):
            user_data = {key: self.get_sequence_slices(value, traj_len) for key, value in user_data.items()}
            uids.append(np.array([user for _ in range(user_data['location'].shape[0])]))
            locations.append(user_data['location'])
            time.append(user_data['time'])
            timeslots.append(user_data['timeslot'])

        self.uid = torch.cat([torch.LongTensor(uid) for uid in uids])
        self.locations = torch.cat([torch.LongTensor(loc) for loc in locations], dim=0)
        self.time = torch.cat([torch.FloatTensor(ts) for ts in time], dim=0)
        self.timeslots = torch.cat([torch.LongTensor(ts) for ts in timeslots], dim=0)
        self.target_len = target_len

    @staticmethod
    def get_sequence_slices(seq, window_size: int):
        # 用来把一个序列切成N个固定长度的子序列
        if not isinstance(seq, np.ndarray):
            seq = np.array(seq)

        if len(seq.shape) > 1:
            seq = seq.T
        shape = seq.shape[:-1] + (seq.shape[-1] - window_size + 1, window_size)
        strides = seq.strides + (seq.strides[-1],)

        if len(seq.shape) > 1:
            return np.lib.stride_tricks.as_strided(seq, shape=shape, strides=strides).transpose(1, 2, 0)

        return np.lib.stride_tricks.as_strided(seq, shape=shape, strides=strides)

    def __getitem__(self, idx):
        # (model's forward), (target)
        return (self.uid[idx], self.locations[idx], self.timeslots[idx]), (
            self.locations[idx][-self.target_len:], self.timeslots[idx][-self.target_len:])

    def __len__(self, ):
        return self.locations.shape[0]


def collate(batch_list):
    # batch_list contains batch_size items
    # each item contain ((uid, traj_loc, traj_tim), (target_loc, target_tim))
    # uid: (); location: (traj_len, ) ; time: (traj_len, )
    # output: uid: (batch_size); location: (B, L); time: (B, L)
    uid = torch.tensor([item[0][0] for item in batch_list])  # (B,)
    loc = torch.stack([item[0][1] for item in batch_list])  # (B, L)
    tim = torch.stack([item[0][2] for item in batch_list])  # (B, L)

    target_loc = torch.stack([item[1][0] for item in batch_list])  # (B, target_len)
    target_tim = torch.stack([item[1][1] for item in batch_list])  # (B, target_len)
    return (uid, loc, tim), (target_loc, target_tim)


def load_data(
        batch_size=DataConfig.batch_size,
        top_locs=DataConfig.top_locs,
        held_out=DataConfig.held_out,
        min_datapoints=DataConfig.min_datapoints,
        traj_len=DataConfig.traj_len,
        target_len=DataConfig.target_len
):
    """
    :param
        set according to the DatasetConfig.
    :return: train_dataloader, valid_dataloader, loc_size, tim_size.(the size of embedding dictionary)
    """
    dyna = preprocess()
    (train_dataset, valid_dataset), user_vocab = sample_dataset(
        data=dyna, top_locs=top_locs, heldout=held_out, min_datapoints=min_datapoints
    )

    print(f'users : {len(user_vocab)}')

    train_data = CheckinDataset(data=train_dataset, traj_len=traj_len, target_len=target_len)
    valid_data = CheckinDataset(data=valid_dataset, traj_len=traj_len, target_len=target_len)

    print(f'items in train dataset : {len(train_data)}')
    print(f'items in valid dataset : {len(valid_data)}')

    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=True, collate_fn=collate, drop_last=True)

    return train_dataloader, valid_dataloader


