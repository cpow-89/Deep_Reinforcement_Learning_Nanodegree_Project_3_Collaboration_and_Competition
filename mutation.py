import copy
import torch
import numpy as np


def gaussian_noise_mutation(parent_network, noise_std, device):
    """Create a mutated copy of the given parent by adding Gaussian noise to the network parameter."""
    child = copy.deepcopy(parent_network)

    for param in child.parameters():
        noise_dist_tensor = torch.tensor(np.random.normal(size=param.data.size()).astype(np.float32)).to(device)
        param.data += noise_std * noise_dist_tensor
    return child
