import numpy as np

import msgpack
import yaml
import os

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
