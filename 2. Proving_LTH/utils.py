import torch
import torch.nn as nn
from torchvision import models



def train_model(model, dataloader, criterion, optimizer, epochs=5):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
                
    return total_loss / len(dataloader)

def train_model_with_mask(model, dataloader, criterion, optimizer, epochs=5, mask=None):
    """Versión modificada que mantiene la máscara durante entrenamiento"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            inputs, targets = inputs.to(device), targets.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            # Aplicar máscara después de cada actualización de parámetros
            if mask is not None:
                with torch.no_grad():
                    for name, param in model.named_parameters():
                        if name in mask:
                            param.data *= mask[name]

            total_loss += loss.item()
                
    return total_loss / len(dataloader)

def evaluate_model(model, dataloader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            correct += (predicted == targets).sum().item()
            total += targets.size(0)
    return correct / total

def create_prune_mask(model, prune_percentage=0.2):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mask = {}
    
    # Recopilar todos los pesos para calcular umbral global
    all_weights = []
    for name, param in model.named_parameters():
        if 'weight' in name and len(param.shape) > 1:
            all_weights.append(param.abs().flatten())
    
    # Concatenar y calcular umbral global
    if all_weights:
        all_weights_tensor = torch.cat(all_weights)
        k = int(prune_percentage * all_weights_tensor.numel())
        if k > 0 and k < all_weights_tensor.numel():
            threshold = torch.kthvalue(all_weights_tensor, k).values
        else:
            threshold = 0.0
        
        # Aplicar umbral a cada capa
        for name, param in model.named_parameters():
            if 'weight' in name and len(param.shape) > 1:
                mask[name] = (param.abs() > threshold).float().to(device)
    
    return mask

def prune_model(model, mask):
    """Aplica la máscara de pruning y registra hooks para mantenerla"""
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in mask:
                param.data *= mask[name]
    
    # Registrar hooks para mantener los pesos podados en cero
    def maintain_mask_hook(grad, mask_tensor):
        return grad * mask_tensor
    
    for name, param in model.named_parameters():
        if name in mask:
            param.register_hook(lambda grad, m=mask[name]: maintain_mask_hook(grad, m))

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.xavier_uniform_(m.weight)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
            
def create_alexnet():
    """Función para crear una nueva instancia de AlexNet"""
    model = models.alexnet(pretrained=False)
    model.classifier[6] = nn.Linear(4096, 10)
    model.apply(init_weights)
    return model