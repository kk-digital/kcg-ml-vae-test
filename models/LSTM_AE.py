import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, h_dim, n_layers, dropout_prob, noise):
        super(Encoder, self).__init__()
        self.noise = noise
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout_prob
        )
        self.fc_latent = nn.Linear(h_dim, latent_dim)

    def forward(self, x):
        if self.noise:
            x = x + torch.randn_like(x) * self.noise
        output, (hidden, cell) = self.lstm(x)
        last_output = hidden[-1, :, :]
        z = self.fc_latent(last_output)

        return z


class Decoder(nn.Module):
    def __init__(self, output_dim, latent_dim, h_dim, out_activ, n_layers, dropout_prob):
        super(Decoder, self).__init__()
        self.output_dim = output_dim
        self.h_dim = h_dim
        self.out_activ = out_activ
        
        self.lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=h_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout_prob
        )
        self.fc_out = nn.Linear(h_dim, output_dim)

    def forward(self, x, seq_len):
        x = x.repeat(seq_len, 1)
        x, (hidden, cell) = self.lstm(x)
        
        x = self.fc_out(x)
        x = x.view(-1, seq_len, self.output_dim)
        if self.out_activ:
            x = self.out_activ(x)

        return x


class LSTMAutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim,
        h_dim,
        n_layers,
        h_activ=nn.Sigmoid(),
        out_activ=nn.Tanh(),
        encoder_dropout_prob=0.0,
        decoder_dropout_prob=0.0,
        noise=0.01
    ):
        super(LSTMAutoEncoder, self).__init__()

        self.encoder = Encoder(input_dim, latent_dim, h_dim, n_layers, encoder_dropout_prob, noise)
        self.decoder = Decoder(input_dim, latent_dim, h_dim, out_activ, n_layers, decoder_dropout_prob)

    def forward(self, x):
        seq_len = x.shape[1]
        self.z = self.encoder(x)
        x = self.decoder(self.z, seq_len)

        return x