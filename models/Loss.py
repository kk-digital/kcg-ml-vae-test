import torch
import torch.nn as nn

class AELoss(nn.Module):
    def __init__(self, lambda_sparsity=0.0, lambda_l1=0.0):
        super(AELoss, self).__init__()
        self.lambda_sparsity = lambda_sparsity
        self.lambda_l1 = lambda_l1

    def _sparsity_loss(self, p_hat, p=0.1):
        p_hat = torch.mean(p_hat, axis=0)
        loss = p * torch.log(p / p_hat) + (1 - p) * torch.log((1 - p) / (1 - p_hat))

        return torch.sum(loss)

    def forward(self, y, y_hat, z, model=None):
        assert y.shape == (y.shape[0], 77, 768)
        assert y_hat.shape == (y.shape[0], 77, 768)

        loss = nn.MSELoss(reduction='sum')(y, y_hat)
        losses = {'mse': loss}
        
        if self.lambda_sparsity:
            # l1_penalty = torch.abs(z).mean()
            sparsity_loss = self.lambda_sparsity * self._sparsity_loss(z)
            loss += sparsity_loss
            losses['sparsity'] = sparsity_loss

        if self.lambda_l1:
            enc_l1_reg = torch.tensor(0.0).to(y.device)
            for param in model.encoder.parameters():
                enc_l1_reg += torch.norm(param, 1)

            dec_l1_reg = torch.tensor(0.0).to(y.device)
            for param in model.decoder.parameters():
                dec_l1_reg += torch.norm(param, 1)

            l1_reg_loss = self.lambda_l1 * (enc_l1_reg + dec_l1_reg)

            loss += l1_reg_loss
            losses['l1'] = l1_reg_loss

        return loss, losses