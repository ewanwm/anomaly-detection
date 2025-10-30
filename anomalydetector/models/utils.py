import typing 

import torch
import numpy as np

from anomalydetector.models.VAE import NFVAE, VAE
from anomalydetector.models.NICE import NICEModel, NICE_gaussian_loss

def get_latent_dists(model, data_loader, device, quiet=False):

    enc_dist_list     = []
    llh_dist_list     = []
    decoded_dist_list = []

    if isinstance(model, NFVAE):
        for x, n in data_loader:
            x = x.to(device)
            encoded, log_q, log_p = model(x, num_samples=1)
            
            # average over sample axis
            encoded = torch.mean(encoded, axis = 1)
            
            decoded_mean, decoded_std = model.decoder(encoded)

            decoded_dist_list.append(decoded_mean.detach().to(torch.device("cpu")))
            enc_dist_list.append(encoded.detach().to(torch.device("cpu")))

    if isinstance(model, VAE):
        for x, n in data_loader:
            x = x.to(device)
            encoded = model.encode(x)
            decoded = model.decode(encoded)
            decoded_dist_list.append(decoded.detach().to(torch.device("cpu")))
            enc_dist_list.append(encoded.detach().to(torch.device("cpu")))


    if isinstance(model, NICEModel):

        for x, n in data_loader:
            x = x.to(device)
            encoded = model(x)
            decoded = model.inverse(encoded)
            enc_dist_list.append(encoded.detach().to(torch.device("cpu")))
            decoded_dist_list.append(decoded.detach().to(torch.device("cpu")))
            llh_dist_list.append(NICE_gaussian_loss(encoded.detach().to(torch.device("cpu")), model.scaling_diag.detach().to(torch.device("cpu")), keepdim=True))

    enc_dist     = np.concatenate(enc_dist_list, axis = 0)
    decoded_dist = np.concatenate(decoded_dist_list, axis = 0)

    llh_dist = None
    if llh_dist_list is []:
        llh_dist_list = np.concatenate(llh_dist_list, axis = 0)
        
    return enc_dist, llh_dist, decoded_dist

def summarise_model(model):

    print("##### Model #####'")
    for module in model.modules():
        print(module)

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])

    print(f"trainable parameters: {params}")
    
