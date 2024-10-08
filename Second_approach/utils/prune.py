import torch
import torch.nn as nn

import numpy as np

def prune_weights(net):
    """This function prunes the 70% of the lowest weights in each layer of the network and sets them to zero"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for name, module in net.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.data.cpu().numpy()
            threshold = np.percentile(np.abs(weights), 70)
            weights[np.abs(weights) < threshold] = 0
            module.weight.data = torch.from_numpy(weights).to(device)
    return net  