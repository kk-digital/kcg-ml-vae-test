import sys
sys.path.append('..')

from utilities.utils import read_embedding_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import TensorDataset

import numpy as np

import os
import torch
import glob
import tqdm

def get_dataset(data_path: str, scaler: str=None) -> (TensorDataset, TensorDataset):
    """Read POSITIVE embeddings from msgpack and create Pytorch dataset

    Args:
        data_path (str): root path to data containing _embedding.msgpack files.
        scaler (str, optional): Scaler to use on data, currently only supports minmax. Defaults to None.

    Returns:
        (TensorDataset, TensorDataset): Train and validation data in PyTorch TensorDataset format.
    """

    # get all files ending with _embedding.msgpack
    embedding_files = []
    for root, dirs, files in os.walk(data_path):
        for file in files:
            if file.endswith('_embedding.msgpack'):
                embedding_files.append(os.path.join(root, file))

    # read the POSTIIVE embeddings and store it in a numpy array
    X = []
    for path in tqdm.tqdm(embedding_files):
        pos_emb, neg_emb = read_embedding_data(path)
        X.append(pos_emb)
    X = np.concatenate(X, axis=0)

    Xtr, Xvl = train_test_split(X, test_size=0.2, random_state=42)

    # scale data. MinMaxScaler fit / transform methods only support 2D
    # thus the need to reshape
    if scaler == 'minmax':
        scl = MinMaxScaler()
        Xtr = Xtr.reshape(-1, 768)
        Xvl = Xvl.reshape(-1, 768)

        Xtr = scl.fit_transform(Xtr)
        Xvl = scl.transform(Xvl)

        # reshape it back
        Xtr = Xtr.reshape(-1, 77, 768)
        Xvl = Xvl.reshape(-1, 77, 768)


    # convert to pytorch dataset format
    train_data = torch.tensor(Xtr, dtype=torch.float32)
    val_data = torch.tensor(Xvl, dtype=torch.float32)

    train_dataset = TensorDataset(train_data)
    val_dataset = TensorDataset(val_data)

    return train_dataset, val_dataset