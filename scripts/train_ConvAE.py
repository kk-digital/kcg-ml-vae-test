import sys
sys.path.append('.')

import torch
import os
import yaml
import pickle
import tqdm

import numpy as np
import torch.nn as nn

from scripts.test_data_generator import generate_test_data
from utilities.train_utils import train_loop, get_device
from utilities.dataloader_embeddings import get_dataset
from models.conv_ae import ConvAE
from models.Loss import AELoss
from torch.utils.data import DataLoader, TensorDataset

configs = {
    'epochs': 5000,
    'learning_rate': 1e-4,
    'batch_size': 128,
    'save_path': './experiments/convae_exp00'
}

os.makedirs(configs['save_path'], exist_ok=True)
sys.stdout = open(os.path.join(configs['save_path'], 'training.log'), 'w')

with open(os.path.join(configs['save_path'], 'configs.yml'), 'w') as f:
    yaml.dump(configs, f, default_flow_style=False)

device = get_device()

train_dataset, val_dataset = get_dataset('./data/', 'standard')

train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False)

model = ConvAE(
    input_dim=768,
    seq_len=77
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=configs['learning_rate'], amsgrad=False)

# criterion = nn.MSELoss(reduction='mean')
criterion = AELoss()

fig, mean_train_loss, mean_val_loss, best_loss = train_loop(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    criterion=criterion,
    device=device,
    optimizer=optimizer,
    epochs=configs['epochs'],
    save_path=configs['save_path'],
    scheduler=None
)