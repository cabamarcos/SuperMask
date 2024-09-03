import torch
import copy

def apply_mask(net, mask):
    # Aplica la máscara a la red: selecciona el 30% de los pesos más altos y desactiva el resto
    net_masked = copy.deepcopy(net)
    with torch.no_grad():
        for net_name, net_param, mask_name, mask_param in zip(net_masked.state_dict(), net_masked.parameters(), mask.state_dict(), mask.parameters()):

            mask_data = mask_param.data.abs()
            threshold = torch.quantile(mask_data, 0.7)
            mask_applied = (mask_data >= threshold).float()

            net_param.data *= mask_applied

    return net_masked
