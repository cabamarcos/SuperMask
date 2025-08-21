import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models

import copy

class MaskedForward(nn.Module):
    def __init__(self, net, mask, sparsity):
        super(MaskedForward, self).__init__()
        self.net = net
        self.mask = mask
        self.sparsity = sparsity

    def _binary_mask(self, scores):
        k = int((1.0 - self.sparsity) * scores.numel())
        threshold = torch.kthvalue(scores.flatten(), k).values
        hard_mask = (scores > threshold).float()
        return hard_mask + (scores - scores.detach())  # STE: binarized forward, gradient passthrough

    def forward(self, x):
        out = x
        for idx, (n_layer, m_layer) in enumerate(zip(self.net.features, self.mask.features)):
            if isinstance(n_layer, nn.Conv2d):
                mask_scores = m_layer.weight.abs()
                mask_bin = self._binary_mask(mask_scores)
                binary_mask = mask_bin + (mask_scores - mask_scores)
                masked_weight = n_layer.weight * binary_mask
                #print(f"[Forward] Conv Layer {idx}: Active weights = {binary_mask.sum().item()} / {binary_mask.numel()}")
                out = F.conv2d(out, masked_weight, n_layer.bias, stride=n_layer.stride, padding=n_layer.padding)
            else:
                out = n_layer(out)

        out = torch.flatten(out, 1)

        for idx, (n_layer, m_layer) in enumerate(zip(self.net.classifier, self.mask.classifier)):
            if isinstance(n_layer, nn.Linear):
                mask_scores = m_layer.weight.abs()
                mask_bin = self._binary_mask(mask_scores)
                binary_mask = mask_bin + (mask_scores - mask_scores)
                masked_weight = n_layer.weight * binary_mask
                #print(f"[Forward] Linear Layer {idx}: Active weights = {binary_mask.sum().item()} / {binary_mask.numel()}")
                out = F.linear(out, masked_weight, n_layer.bias)
            else:
                out = n_layer(out)

        return out

def restart_training(device):
    """Función para reinicializar net, mask y métricas"""
    global net, mask, optimizer, epoch, losses, accuracies, saved, restart_count
    
    restart_count += 1
    print("Reinicializando net y máscara, y reiniciando el entrenamiento...")
    
    net = models.alexnet(pretrained=False)
    net.classifier[6] = nn.Linear(net.classifier[6].in_features, 10)
    net.to(device)
    
    # Congelar los parámetros de net
    for param in net.parameters():
        param.requires_grad = False
    
    mask = models.alexnet(pretrained=False)
    mask.classifier[6] = nn.Linear(mask.classifier[6].in_features, 10)
    mask.to(device)
    
    optimizer = optim.Adam(mask.parameters(), lr=0.001)
    
    epoch = 0
    losses = []
    accuracies = []
    saved = False

def check_restart_conditions(restart_checks, device):
    """Verifica todas las condiciones de reinicio usando el diccionario de configuración"""
    current_max_accuracy = max(accuracies) if accuracies else 0
    
    for check_epoch, config in restart_checks.items():
        min_acc = config["accuracy"] * 100
        restart_type = config["message"]
        
        # Para época 70, verificar si es mayor o igual; para otras, verificar igualdad exacta
        should_check = (epoch == check_epoch) or (check_epoch == 70 and epoch >= check_epoch)
        
        if should_check and current_max_accuracy < min_acc:
            print(f"\n!!! {restart_type} {restart_count + 1} !!!")
            print(f"No se alcanzó {min_acc}% de accuracy en época {check_epoch}.")
            restart_training(device)
            return True
    
    return False