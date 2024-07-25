import numpy as np

def apply_mask(net, mask, percentage=30):
    # Obtiene los pesos de ambos modelos
    mask_weights = mask.get_weights()
    net_weights = net.get_weights()

    # Procesa cada capa
    new_weights = []
    for mw, nw in zip(mask_weights, net_weights):
        # Aplana los pesos para facilitar la manipulación
        mw_flat = mw.flatten()

        # Determina el umbral para el porcentaje dado
        threshold = np.percentile(mw_flat, 100 - percentage)

        # Crea la máscara con 1s para el porcentaje más alto y 0s para el porcentaje más bajo
        mask = np.where(mw >= threshold, 1, 0)

        # Aplica la máscara a los pesos de net
        new_w = nw * mask

        # Reshapea los pesos al formato original
        new_weights.append(new_w)

    # Asigna los nuevos pesos al modelo net
    net.set_weights(new_weights)