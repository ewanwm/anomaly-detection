import typing 

import torch
import numpy as np

from anomalydetector.models.VAE import NFVAE, VAE
from anomalydetector.models.NICE import NICEModel, NICE_gaussian_loss

def get_latent_dists(model, data_loader, device, quiet=False, num_samples:typing.Optional[int] = None):
    
    n_features = data_loader.dataset.get_n_features()

    enc_dist = np.ndarray((1, n_features))
    llh_dist = np.ndarray((1, 1))

    if isinstance(model, NFVAE):
        for x, n in data_loader:
            x = x.to(device)
            encoded, log_q, log_p = model(x, num_samples)
            decoded = model.decoder(encoded)
            enc_dist = np.concatenate((enc_dist, encoded.detach().to(torch.device("cpu"))), axis=0)


    if isinstance(model, VAE):
        for x, n in data_loader:
            x = x.to(device)
            encoded = model.encode(x)
            enc_dist = np.concatenate((enc_dist, encoded.detach().to(torch.device("cpu"))), axis=0)


    if isinstance(model, NICEModel):

        for x, n in data_loader:
            x = x.to(device)
            encoded = model(x)
            enc_dist = np.concatenate((enc_dist, encoded.detach().to(torch.device("cpu"))), axis=0)
            llh_dist = np.concatenate((llh_dist, NICE_gaussian_loss(encoded.detach().to(torch.device("cpu")), model.scaling_diag.detach().to(torch.device("cpu")), keepdim=True)), axis=0)

    return enc_dist, llh_dist

def summarise_model(model):

    print("##### Model #####'")
    for module in model.modules():
        print(module)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"trainable parameters: {params}")
    