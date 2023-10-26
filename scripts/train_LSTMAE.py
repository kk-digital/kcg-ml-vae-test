import sys
sys.path.append('.')

import torch
import os
import yaml

import torch.nn as nn

from scripts.test_data_generator import generate_test_data
from utilities.train_utils import train_loop, get_device
from utilities.dataloader_embeddings import get_dataset
from models.LSTM_AE import LSTMAutoEncoder
from models.Loss import AELoss
from torch.utils.data import DataLoader, TensorDataset

configs = {
    'input_dim': 768,
    'latent_dim': 768,
    'h_dim': 1024,
    'n_layers': 3,
    'out_activ': nn.Tanh(),
    'encoder_dropout_prob': 0.5,
    'decoder_dropout_prob': 0.0,

    'noise': 0.0,
    
    'epochs': 20,
    'learning_rate': 1e-4,
    'batch_size': 32,
    'save_path': './experiments/lstm_syn_exp00'
}

os.makedirs(configs['save_path'], exist_ok=True)
sys.stdout = open(os.path.join(configs['save_path'], 'training.log'), 'w')

with open(os.path.join(configs['save_path'], 'configs.yml'), 'w') as f:
    yaml.dump(configs, f, default_flow_style=False)

device = get_device()

# generate and load synthetic data
train_data = generate_test_data(n_samples=2000, data_dim=768, sequence_len=77, drift_step_size=0.01, mode='clip')
val_data = generate_test_data(n_samples=200, data_dim=768, sequence_len=77, drift_step_size=0.01, mode='clip')

train_data = torch.tensor(train_data, dtype=torch.float32)
val_data = torch.tensor(val_data, dtype=torch.float32)

train_dataset = TensorDataset(train_data)
val_dataset = TensorDataset(val_data)

# train_dataset, val_dataset = get_dataset('./data/', 'minmax')

train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False)

model = LSTMAutoEncoder(
    input_dim=configs['input_dim'],
    latent_dim=configs['latent_dim'],
    h_dim=configs['h_dim'],
    n_layers=configs['n_layers'],
    out_activ=configs['out_activ'],
    encoder_dropout_prob=configs['encoder_dropout_prob'],
    decoder_dropout_prob=configs['decoder_dropout_prob'],
    noise=configs['noise']
)
optimizer = torch.optim.AdamW(model.parameters(), lr=configs['learning_rate'])

model = model.to(device)
criterion = AELoss()

print(f'\nTraining Start')
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

print('\nTraining Completed')
print(f'Best Val Loss: {best_loss}')