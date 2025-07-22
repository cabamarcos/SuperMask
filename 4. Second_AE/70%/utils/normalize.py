import torch
import torch.nn as nn

# Función general para normalizar los pesos de una capa en cualquier rango [a, b]
def normalize_weights(layer, lower_bound=0.0, upper_bound=1.0):
    with torch.no_grad():  # Desactivamos el gradiente porque estamos actualizando los pesos
        # Normalizamos los pesos
        weights = layer.weight.data
        w_min = weights.min()
        w_max = weights.max()
        
        normalized_weights = lower_bound + (weights - w_min) / (w_max - w_min) * (upper_bound - lower_bound)
        layer.weight.data = normalized_weights
        
        # Normalizamos los biases (si la capa tiene bias)
        if layer.bias is not None:
            biases = layer.bias.data
            b_min = biases.min()
            b_max = biases.max()
            
            normalized_biases = lower_bound + (biases - b_min) / (b_max - b_min) * (upper_bound - lower_bound)
            layer.bias.data = normalized_biases
