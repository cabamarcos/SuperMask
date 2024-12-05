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

    # # Calcula los porcentajes
    # percentage_ones = (total_ones / total_elements) * 100
    # percentage_zeros = 100 - percentage_ones

    # # Muestra el resultado
    # print(f"Porcentaje de 1s: {percentage_ones:.2f}%")
    # print(f"Porcentaje de 0s: {percentage_zeros:.2f}%")
    
    return individuo


def modify_weights(network):
    """
    Modifica aleatoriamente el 5% de los pesos de 1 a 0 y el 5% de los pesos de 0 a 1,
    manteniendo el 30% de 1s y el 70% de 0s en la red.

    Args:
        network (torch.nn.Module): Red con pesos que consisten en 1s y 0s.

    Returns:
        torch.nn.Module: Red modificada con los cambios aplicados.
    """
    # Convertir los pesos a un tensor plano
    with torch.no_grad():
        weights = torch.flatten(torch.cat([p.flatten() for p in network.parameters()]))

    # Verificar distribución inicial
    ones_indices = (weights == 1).nonzero(as_tuple=True)[0]
    zeros_indices = (weights == 0).nonzero(as_tuple=True)[0]

    total_weights = weights.numel()
    total_ones = len(ones_indices)
    total_zeros = len(zeros_indices)

    assert abs(total_ones / total_weights - 0.3) < 0.01, "La red inicial no tiene aproximadamente el 30% de pesos = 1."
    assert abs(total_zeros / total_weights - 0.7) < 0.01, "La red inicial no tiene aproximadamente el 70% de pesos = 0."

    # Calcular el número exacto de cambios necesarios para mantener la proporción
    num_to_flip_1_to_0 = int(0.05 * total_weights)
    num_to_flip_0_to_1 = int(0.05 * total_weights)

    # Seleccionar índices aleatorios para los cambios
    flip_1_to_0_indices = random.sample(list(ones_indices.cpu().numpy()), num_to_flip_1_to_0)
    flip_0_to_1_indices = random.sample(list(zeros_indices.cpu().numpy()), num_to_flip_0_to_1)

    # Realizar los cambios
    weights[flip_1_to_0_indices] = 0
    weights[flip_0_to_1_indices] = 1

    # Ajustar la proporción si es necesario
    # Recalcular índices después de los cambios
    ones_indices = (weights == 1).nonzero(as_tuple=True)[0]
    zeros_indices = (weights == 0).nonzero(as_tuple=True)[0]

    total_ones = len(ones_indices)
    total_zeros = len(zeros_indices)

    target_ones = int(0.3 * total_weights)
    target_zeros = total_weights - target_ones

    if total_ones > target_ones:
        # Demasiados 1s: convertir el exceso a 0s
        excess_ones = total_ones - target_ones
        excess_indices = random.sample(list(ones_indices.cpu().numpy()), excess_ones)
        weights[excess_indices] = 0
    elif total_zeros > target_zeros:
        # Demasiados 0s: convertir el exceso a 1s
        excess_zeros = total_zeros - target_zeros
        excess_indices = random.sample(list(zeros_indices.cpu().numpy()), excess_zeros)
        weights[excess_indices] = 1

    # Restaurar los pesos modificados en la red
    current_idx = 0
    for param in network.parameters():
        numel = param.numel()
        param.data.copy_(weights[current_idx:current_idx + numel].view_as(param))
        current_idx += numel


    # #imprimir porcentaje de 1s y 0s
    # ones_indices = (weights == 1).nonzero(as_tuple=True)[0]
    # zeros_indices = (weights == 0).nonzero(as_tuple=True)[0]

    # total_weights = weights.numel()
    # total_ones = len(ones_indices)
    # total_zeros = len(zeros_indices)

    # print(f"Porcentaje de 1s: {total_ones / total_weights * 100:.2f}%")
    # print(f"Porcentaje de 0s: {total_zeros / total_weights * 100:.2f}%")

    return network