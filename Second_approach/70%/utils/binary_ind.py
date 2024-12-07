import random
import torch

def make_to_binary(individuo):
    total_elements = 0
    total_ones = 0

    for name, param in individuo.named_parameters():
        if "weight" in name or "bias" in name:
            # Calcula el percentil 30 directamente sobre los valores de los parámetros
            threshold = param.quantile(0.7)

            # Binariza: 1 para valores mayores al percentil 30, 0 para los demás
            binary_param = torch.where(param > threshold, 
                                       torch.tensor(1, dtype=torch.float).to(param.device), 
                                       torch.tensor(0, dtype=torch.float).to(param.device))
            
            # Actualiza los valores binarizados en los parámetros
            param.data = binary_param
            
            # Actualiza contadores
            total_elements += binary_param.numel()
            total_ones += (binary_param == 1).sum().item()
    
    return individuo


def modify_weights(network):
    """
    Modifica la red asegurando que los pesos binarizados siempre mantengan un 10% de activación.
    
    Args:
        network (torch.nn.Module): Red con pesos binarizados (1s y 0s).
        
    Returns:
        torch.nn.Module: Red modificada con exactamente el 10% de activación.
    """
    with torch.no_grad():
        # Convertir los pesos a un tensor plano
        weights = torch.cat([param.flatten() for param in network.parameters()]).clone()

        # Total de pesos y objetivo del 30%
        total_weights = weights.numel()
        target_ones = int(0.3 * total_weights)  # Queremos exactamente el 30%

        # Identificar índices de unos y ceros
        ones_indices = (weights == 1).nonzero(as_tuple=True)[0]
        zeros_indices = (weights == 0).nonzero(as_tuple=True)[0]

        # Modificar aleatoriamente un 5% de los pesos
        num_to_flip_1_to_0 = min(len(ones_indices), int(0.05 * total_weights))
        num_to_flip_0_to_1 = min(len(zeros_indices), int(0.05 * total_weights))

        # Convertir 1s a 0s y 0s a 1s
        if num_to_flip_1_to_0 > 0:
            flip_1_to_0_indices = random.sample(list(ones_indices.cpu().numpy()), num_to_flip_1_to_0)
            weights[flip_1_to_0_indices] = 0
        if num_to_flip_0_to_1 > 0:
            flip_0_to_1_indices = random.sample(list(zeros_indices.cpu().numpy()), num_to_flip_0_to_1)
            weights[flip_0_to_1_indices] = 1

        # Ajustar para garantizar exactamente el 30% de activación
        ones_indices = (weights == 1).nonzero(as_tuple=True)[0]
        zeros_indices = (weights == 0).nonzero(as_tuple=True)[0]

        # Calcular exceso o déficit de unos
        current_ones = len(ones_indices)
        excess_ones = current_ones - target_ones

        if excess_ones > 0:
            # Demasiados 1s: convertir el exceso a 0s
            flip_indices = random.sample(list(ones_indices.cpu().numpy()), excess_ones)
            weights[flip_indices] = 0
        elif excess_ones < 0:
            # Faltan 1s: convertir ceros adicionales a 1s
            flip_indices = random.sample(list(zeros_indices.cpu().numpy()), -excess_ones)
            weights[flip_indices] = 1

        # Restaurar los pesos modificados en la red
        current_idx = 0
        for param in network.parameters():
            numel = param.numel()
            param.data.copy_(weights[current_idx:current_idx + numel].view_as(param))
            current_idx += numel

    return network

def apply_mask_binary(net, individuo):
    """
    Modifica los pesos de `net` usando una máscara basada en los pesos de `individuo`.
    
    Args:
        net: Red neuronal en PyTorch cuyas conexiones serán ajustadas.
        individuo: Red neuronal en PyTorch que se usará para calcular la máscara.
    
    Returns:
        net: Red neuronal modificada según la máscara.
    """
    for (param_net, param_individuo) in zip(net.parameters(), individuo.parameters()):
        # Asumimos que `param_individuo` es binario, donde 1 indica un peso activo y 0 un peso inactivo.
        # Crea la máscara directamente a partir de `param_individuo`.
        mascara = param_individuo.data.clone()

        # Aplica la máscara a los pesos de `net`.
        param_net.data = param_net.data * mascara

    return net
