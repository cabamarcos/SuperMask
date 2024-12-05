import torch
import torch.nn as nn

import numpy as np
import copy

def prune_weights(net):
    """This function prunes the 70% of the lowest weights in each layer of the network and sets them to zero"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    new_net = copy.deepcopy(net)  # Create a copy of the network
    for name, module in new_net.named_modules():
        if isinstance(module, nn.Linear):
            weights = module.weight.data.cpu().numpy()
            threshold = np.percentile(np.abs(weights), 70)
            weights[np.abs(weights) < threshold] = 0
            module.weight.data = torch.from_numpy(weights).to(device)
    return new_net

def apply_mask(net, individuo):
    """
    Modifica los pesos de `net` usando una máscara basada en los pesos de `individuo`.
    
    Args:
        net: Red neuronal en PyTorch cuyas conexiones serán ajustadas.
        individuo: Red neuronal en PyTorch que se usará para calcular la máscara.
    
    Returns:
        net: Red neuronal modificada según la máscara.
    """
    for (param_net, param_individuo) in zip(net.parameters(), individuo.parameters()):
        # Obtiene los pesos de `individuo` en forma de un tensor
        pesos_individuo = param_individuo.data.clone()
        
        # Calcula el percentil 70 para determinar el umbral
        percent = torch.quantile(pesos_individuo.abs().flatten(), 0.9)
        
        # Crea la máscara: 1 donde el valor >= percentil_70, 0 donde < percentil_70
        mascara = (pesos_individuo.abs() >= percent).float()

        param_net.data = param_net.data * mascara

    return net
