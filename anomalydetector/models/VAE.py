import typing

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

        self.loss_fn = MSELoss()

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
        loss = self.loss_fn(data, rc_batch)
        loss.backward()

        return loss
    
class NFVAE(nf.NormalizingFlowVAE):
    
    def __init__(
            self, 
            n_bottleneck:int, 
            hidden_units_encoder:typing.List[int], 
            hidden_units_decoder:typing.List[int], 
            n_flows:int, 
            flow_type:str, 
            num_samples:int,
            device
        ):

        ## create normal distribution to use as prior
        self.prior = torch.distributions.MultivariateNormal(
            torch.zeros(n_bottleneck, device=device),
            torch.eye(n_bottleneck, device=device)
        )
        
        ## set up encoder and decoder networks
        encoder_nn = nf.nets.MLP(hidden_units_encoder)
        decoder_nn = nf.nets.MLP(hidden_units_decoder)
        
        encoder = nf.distributions.NNDiagGaussian(encoder_nn)
        decoder = nf.distributions.NNBernoulliDecoder(decoder_nn)

        ## set up the flows
        flows = None

        if flow_type == 'Planar':
            
            flows = [nf.flows.Planar((n_bottleneck,)) for k in range(n_flows)]
        
        elif flow_type == 'Radial':
            
            flows = [nf.flows.Radial((n_bottleneck,)) for k in range(n_flows)]
        
        elif flow_type == 'RealNVP':
        
            b = torch.tensor(n_bottleneck // 2 * [0, 1] + n_bottleneck % 2 * [0])
            
            flows = []
            
            for i in range(n_flows):
                s = nf.nets.MLP([n_bottleneck, n_bottleneck])
                t = nf.nets.MLP([n_bottleneck, n_bottleneck])
                if i % 2 == 0:
                    flows += [nf.flows.MaskedAffineFlow(b, t, s)]
                else:
                    flows += [nf.flows.MaskedAffineFlow(1 - b, t, s)]
        else:
            raise NotImplementedError

        ## instantiate the nf.NormalizingFlowVAE
        super().__init__(self.prior, encoder, flows, decoder)

        self.num_samples = num_samples

    def set_num_samples(self, num_samples):
        
        self.num_samples = num_samples

    def train_batch(self, data):

        z, log_q, log_p = self(data, self.num_samples)

        mean_log_q = torch.mean(log_q)
        mean_log_p = torch.mean(log_p)
        
        loss = mean_log_q - mean_log_p
        
        loss.backward()

        return loss
