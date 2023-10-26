import torch.nn as nn
import torch.nn.functional as F
import torch

class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling 
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf
    """
    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)
        
    def forward(self, batch_rep):
        """
        input:
            batch_rep : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension
        
        attention_weight:
            att_w : size (N, T, 1)
        
        return:
            utter_rep: size (N, H)
        """
        softmax = nn.functional.softmax
        att_w = softmax(self.W(batch_rep).squeeze(-1)).unsqueeze(-1)
        utter_rep = torch.sum(batch_rep * att_w, dim=1)

        return utter_rep

class Encoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_units, dropout_prob, noise=0.0, sparsity=False, pooling='max'):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_units = hidden_units
        self.noise = noise
        self.sparsity = sparsity

        self.layer_norm = nn.LayerNorm(self.input_dim)
        
        # define encoder layers: 3 layers fc of 1024, last fc no activation
        encoder_layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                encoder_layers.append(nn.Linear(self.input_dim, hidden_units[i]))
            else:
                encoder_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            encoder_layers.append(nn.ReLU())
            encoder_layers.append(nn.Dropout(dropout_prob))
        self.encoder_layers = nn.Sequential(*encoder_layers)

        reduction_layers = [
            nn.Conv1d(
                in_channels=hidden_units[-1],
                out_channels=32,
                kernel_size=5,
                padding='same',
                padding_mode='circular'
            ),
            nn.Dropout(dropout_prob)
        ]
        self.reduction_layers = nn.Sequential(*reduction_layers)
        self.attention_pooling = SelfAttentionPooling(hidden_units[-1])
        self.fc_latent = nn.Linear(hidden_units[-1], self.latent_dim)

    def forward(self, x):
        assert x.shape == (x.shape[0], 77, 768)

        if self.noise:
            x = x + torch.randn_like(x) * self.noise

        # reshape for fc
        x = x.view(-1, 768)
        x = self.encoder_layers(x)

        # reshape back to batch_size * 77 x hidden size
        x = x.view(-1, 77, self.hidden_units[-1])
        # x = self.reduction_layers(x.permute(0, 2, 1))
        x = self.attention_pooling(x)
        print(x.shape)
        # x = x.view(-1, 77 * 32)
        x = self.fc_latent(x)
        if self.sparsity:
            x = nn.Sigmoid()(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, input_dim, latent_dim, hidden_units, dropout_prob, out_act='tanh'):
        super(Decoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # define decoder layers: 3 layers of fc 1024, last layer activated with tanh
        decoder_layers = []
        for i in range(len(hidden_units)):
            if i == 0:
                decoder_layers.append(nn.Linear(self.latent_dim, hidden_units[i]))
            else:
                decoder_layers.append(nn.Linear(hidden_units[i-1], hidden_units[i]))
            decoder_layers.append(nn.ReLU())
            decoder_layers.append(nn.Dropout(dropout_prob))

        decoder_layers.append(nn.Linear(hidden_units[-1], 77 * self.input_dim))
        self.decoder_layers = nn.Sequential(*decoder_layers)

        if out_act == 'tanh':
            self.out_act = nn.Tanh()
        elif out_act == 'sigmoid':
            self.out_act = nn.Sigmoid()
        elif out_act == None:
            self.out_act = None

    def forward(self, z):
        assert z.shape == (z.shape[0], self.latent_dim)

        x = self.decoder_layers(z)
        if self.out_act:
            x = self.out_act(x)

        # reshape to batch_size * 77 * 768
        x = x.view(-1, 77, 768)

        return x
    
class AutoEncoder(nn.Module):
    def __init__(
        self,
        input_dim,
        latent_dim, 
        encoder_hidden_units,
        decoder_hidden_units,
        encoder_dropout_prob,
        decoder_dropout_prob,
        pooling='max',
        out_act='tanh',
        noise=0.0,
        sparsity=False,
        codebook_size=0
    ):
        super(AutoEncoder, self).__init__()

        self.encoder = Encoder(input_dim, latent_dim, encoder_hidden_units, encoder_dropout_prob, noise, sparsity, pooling)
        self.decoder = Decoder(input_dim, latent_dim, decoder_hidden_units, decoder_dropout_prob, out_act)

        if codebook_size:
            self.codebook = nn.Embedding(codebook_size, latent_dim)

    def _quantize(self, latent_representation):
        # Calculate the distance between the latent representation and each codebook vector
        distances = torch.cdist(latent_representation, self.codebook.weight, p=2)

        # Find the codebook vector that is closest to the latent representation
        nearest_codebook_indices = torch.argmin(distances, dim=1)

        # Return the quantized latent representation
        quantized_latent_representation = self.codebook(nearest_codebook_indices)

        return quantized_latent_representation

    def forward(self, x):
        self.z = self.encoder(x)
        recon = self.decoder(self.z)

        return recon
    
