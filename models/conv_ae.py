from turtle import forward
import torch
import torch.nn as nn

class Encoder1DCNN(nn.Module):
    def __init__(self, seq_len=77, input_dim=768, latent_dim=4096):
        super(Encoder1DCNN, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.encoder_layers = nn.Sequential(
            nn.Conv1d(self.input_dim, 4096, kernel_size=3, stride=1, dilation=2, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(4096, 2048, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(2048, 2048, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(2048, 1024, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(1024, 1024, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(1024, 512, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(512, 512, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(512, 256, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU(),
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding='same', padding_mode='circular'),
            nn.ReLU()
        )

        # Skip connections
        for i in range(1, len(self.encoder_layers), 2):
            self.encoder_layers[i].add_module('skip', nn.Identity())

        self.fc_latent = nn.Linear(self.seq_len * 256, latent_dim)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.encoder_layers(x)
        x = x.view(x.size(0), -1)

        output = self.fc_latent(x)

        return output
    
class Decoder1DCNN(nn.Module):
    def __init__(self, seq_len=77, input_dim=768, latent_dim=4096):
        super(Decoder1DCNN, self).__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.fc = nn.Linear(latent_dim, self.seq_len * 512)
        self.decoder_layers = nn.Sequential(
            nn.ConvTranspose1d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(1024, 2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(2048, 2048, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(2048, self.input_dim, kernel_size=3, stride=1, padding=1)
        )

        # Residual connections
        for i in range(1, len(self.decoder_layers), 2):
            self.decoder_layers[i].add_module('skip', nn.Identity())

    def forward(self, z):
        x = self.fc(z)
        x = x.view(x.size(0), 512, self.seq_len)
        x = self.decoder_layers(x)
        x = x.permute(0, 2, 1)

        return x
    
class ConvAE(nn.Module):
    def __init__(self, input_dim, seq_len):
        super(ConvAE, self).__init__()

        self.encoder = Encoder1DCNN(input_dim=input_dim, seq_len=seq_len)
        self.decoder = Decoder1DCNN(input_dim=input_dim, seq_len=seq_len)

    def forward(self, x):
        self.z = self.encoder(x)
        x = self.decoder(self.z)

        return x