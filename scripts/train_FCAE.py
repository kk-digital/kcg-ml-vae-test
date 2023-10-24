import sys
sys.path.append('.')

import torch
import os
import yaml

from utilities.utils import get_dataset
from utilities.train_utils import train_loop, get_device
from models.FC_AE import AutoEncoder, LossFC
from torch.utils.data import DataLoader

configs = {
    'learning_rate': 1e-3,
    'pooling': 'avg',
    'batch_size': 128,
    'epoch': 2000,
    'latent_dim': 768,
    'encoder_hidden_units': [2048] * 5,
    'decoder_hidden_units': [2048] * 5,
    'encoder_dropout_prob': 0.5,
    'decoder_dropout_prob': 0.0,
    'save_path': './experiments/fc_exp10',
    'weight_decay': 0.1,
    'lambda_sparsity': 0.1,
    'lambda_l1': 0.1,
    'noise': 0.0,
    'codebook_size': 0
}

os.makedirs(configs['save_path'], exist_ok=True)
sys.stdout = open(os.path.join(configs['save_path'], 'training.log'), 'w')

with open(os.path.join(configs['save_path'], 'configs.yml'), 'w') as f:
    yaml.dump(configs, f, default_flow_style=False)

device = get_device()

train_dataset, val_dataset = get_dataset()
train_loader = DataLoader(train_dataset, batch_size=configs['batch_size'], shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=configs['batch_size'], shuffle=False)

model = AutoEncoder(
    input_dim=768,
    latent_dim=configs['latent_dim'],
    pooling=configs['pooling'],
    out_act='sigmoid',
    encoder_hidden_units=configs['encoder_hidden_units'],
    decoder_hidden_units=configs['decoder_hidden_units'],
    encoder_dropout_prob=configs['encoder_dropout_prob'],
    decoder_dropout_prob=configs['decoder_dropout_prob'],
    noise=configs['noise'],
    codebook_size=configs['codebook_size']
)
# optimizer = torch.optim.SGD(model.parameters(), lr=configs['learning_rate'], momentum=0.9, weight_decay=configs['weight_decay'])
optimizer = torch.optim.AdamW(model.parameters(), lr=configs['learning_rate'])
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, threshold=0.001)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_loader), epochs=configs['epoch'])
model = model.to(device)

criterion = LossFC(
    lambda_l1=configs['lambda_l1'],
    lambda_sparsity=configs['lambda_sparsity']
)
print(f'\nTraining Start')
fig, mean_train_loss, mean_val_loss, best_loss = train_loop(
    train_loader=train_loader,
    val_loader=val_loader,
    model=model,
    criterion=criterion,
    device=device,
    optimizer=optimizer,
    epochs=configs['epoch'],
    save_path=configs['save_path'],
    scheduler=scheduler
)

print('\nTraining Completed')
print(f'Best Val Loss: {best_loss}')