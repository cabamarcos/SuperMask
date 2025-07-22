import torch
import numpy as np

def calculate_active_weights_percentage(model):
    """
    Calcula la cantidad de pesos y biases activos (!= 0) en un modelo de PyTorch.

    Args:
        model: Modelo de PyTorch.

    Returns:
        Un diccionario con el número total de parámetros, la cantidad de parámetros activos,
        y el porcentaje de parámetros activos.
    """
    total_params = 0
    active_params = 0

    with torch.no_grad():
        for param in model.parameters():
            total_params += param.numel()
            active_params += (param != 0).sum().item()

    active_percentage = (active_params / total_params) * 100 if total_params > 0 else 0

    return {
        "total_params": total_params,
        "active_params": active_params,
        "active_percentage": active_percentage
    }

