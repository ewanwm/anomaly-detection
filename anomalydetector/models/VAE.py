import normflows as nf

import torch
from torch import nn
from torch.distributions.normal import Normal
from torch.nn import MSELoss

class VAE(nn.Module):
    def __init__(self, n_features, latent_size):
        super().__init__()

        self._n_features = n_features
        self._latent_size = latent_size

        self.encode = nn.Sequential(
            nn.Linear(self._n_features, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )

        self.f1 = nn.Linear(256, self._latent_size)
        self.f2 = nn.Linear(256, self._latent_size)
        
        self.decode = nn.Sequential(
            nn.Linear(self._latent_size, 256),
            nn.ReLU(True),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Linear(512, self._n_features),
        )

    def forward(self, x):
        # Encode

        mu = self.f1( self.encode(x) )
        log_var = self.f2(self.encode(x))

        # Reparametrize variables
        std = torch.exp(0.5 * log_var)
        norm_scale = torch.randn_like(std) + 0.00001
        z_ = mu + norm_scale * std

        # Q0 and prior
        q0 = Normal(mu, torch.exp((0.5 * log_var)))
        p = Normal(0.0, 1.0)

        # Decode
        z_ = z_.view(z_.size(0), self._latent_size)
        zD = self.decode(z_)
        out = zD

        return out, mu, log_var
    
    def train_batch(self, data):

        rc_batch, mu, log_var = self.forward(data)
        loss = MSELoss()(data, rc_batch)
        loss.backward()

        return loss
    
class NFVae(nf.NormalizingFlowVAE):
    
    def fit_batch(self, data):

        z, log_q, log_p = self(data, self.num_samples)
        mean_log_q = torch.mean(log_q)
        mean_log_p = torch.mean(log_p)
        loss = mean_log_q - mean_log_p
        loss.backward()

        return loss
