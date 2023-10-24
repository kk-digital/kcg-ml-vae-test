import numpy as np

import msgpack
import yaml
import os
import glob
import tqdm
import torch

from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from mega import Mega

def read_yaml(file_path):
    with open(file_path, 'rb') as f:
        data = yaml.load(f, Loader=yaml.SafeLoader)

    return data

def read_msg_pack(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    decoded_data = msgpack.unpackb(data)

    return decoded_data

def read_embedding_data(file_path):
    # TODO: may need to move this to a better location
    decoded_data = read_msg_pack(file_path)
    pos_emb = np.array(decoded_data['positive_embedding']['__ndarray__'])
    neg_emb = np.array(decoded_data['negative_embedding']['__ndarray__'])

    return pos_emb, neg_emb

def download_mega(src_url, output_path):
    mega = Mega()

    if type(src_url) == list and type(output_path) == list:
        for url, output in zip(src_url, output_path):
            os.makedirs(output, exist_ok=True)
            mega.download_url(url, output)
    else:
        os.makedirs(output_path, exist_ok=True)
        mega.download_url(src_url, output_path)

    return output_path

def get_dataset():
    paths = sorted(glob.glob('./data/000*/*_embedding.msgpack'))

    X = []
    for path in tqdm.tqdm(paths):
        pos_emb, neg_emb = read_embedding_data(path)
        X.append(pos_emb)
    X = np.concatenate(X, axis=0)

    scl = MinMaxScaler()

    Xtr, Xvl = train_test_split(X, test_size=0.2, random_state=42)

    Xtr = Xtr.reshape(-1, 768)
    Xvl = Xvl.reshape(-1, 768)

    Xtr = scl.fit_transform(Xtr)
    Xvl = scl.transform(Xvl)

    # reshape it back
    Xtr = Xtr.reshape(-1, 77, 768)
    Xvl = Xvl.reshape(-1, 77, 768)

    train_data = torch.tensor(Xtr, dtype=torch.float32)
    val_data = torch.tensor(Xvl, dtype=torch.float32)

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    return train_dataset, val_dataset
