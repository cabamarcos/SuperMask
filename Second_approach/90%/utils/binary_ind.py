import random
import torch

def make_to_binary(individuo):
    total_elements = 0
    total_ones = 0

    for name, param in individuo.named_parameters():
        if "weight" in name or "bias" in name:
            # Calcula el percentil 10 directamente sobre los valores de los parámetros
            threshold = param.quantile(0.9)

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
    Modifica la red asegurando que los pesos binarizados siempre mantengan un 10% de activación.
    
    Args:
        network (torch.nn.Module): Red con pesos binarizados (1s y 0s).
        
    Returns:
        torch.nn.Module: Red modificada con exactamente el 10% de activación.
    """
    with torch.no_grad():
        # Convertir los pesos a un tensor plano
        weights = torch.cat([param.flatten() for param in network.parameters()]).clone()

        # Total de pesos y objetivo del 10%
        total_weights = weights.numel()
        target_ones = int(0.1 * total_weights)  # Queremos exactamente el 10%

        # Identificar índices de unos y ceros
        ones_indices = (weights == 1).nonzero(as_tuple=True)[0]
        zeros_indices = (weights == 0).nonzero(as_tuple=True)[0]

        # Modificar aleatoriamente un 2% de los pesos
        num_to_flip_1_to_0 = min(len(ones_indices), int(0.02 * total_weights))
        num_to_flip_0_to_1 = min(len(zeros_indices), int(0.02 * total_weights))

        # Convertir 1s a 0s y 0s a 1s
        if num_to_flip_1_to_0 > 0:
            flip_1_to_0_indices = random.sample(list(ones_indices.cpu().numpy()), num_to_flip_1_to_0)
            weights[flip_1_to_0_indices] = 0
        if num_to_flip_0_to_1 > 0:
            flip_0_to_1_indices = random.sample(list(zeros_indices.cpu().numpy()), num_to_flip_0_to_1)
            weights[flip_0_to_1_indices] = 1

        # Ajustar para garantizar exactamente el 10% de activación
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


import torch
import random

# Red de prueba
class TestNetwork(torch.nn.Module):
    def __init__(self, size=1000):
        super(TestNetwork, self).__init__()
        self.layer1 = torch.nn.Linear(size, size)
        self.layer2 = torch.nn.Linear(size, size)

network = TestNetwork()

# Función para verificar proporción de 1s
def check_proportion(network, target_percentage=10.0):
    total_params = 0
    active_params = 0
    with torch.no_grad():
        for param in network.parameters():
            total_params += param.numel()
            active_params += (param == 1).sum().item()
    active_percentage = (active_params / total_params) * 100
    return active_percentage

# Bucle para probar
iterations = 10
print("Proporciones iniciales de 1s antes de modificar:")
print(f"Iteración inicial: {check_proportion(network):.2f}% de pesos activos")

for i in range(1, iterations + 1):
    # Inicializar red
    network = TestNetwork()
    network = make_to_binary(network)  # Convertimos a binario
    network = modify_weights(network)  # Aplicamos la función de modificación
    active_percentage = check_proportion(network)
    print(f"Iteración {i}: {active_percentage:.2f}% de pesos activos")
    assert abs(active_percentage - 10.0) < 0.01, f"Error: proporción fuera del rango permitido en la iteración {i}"


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
