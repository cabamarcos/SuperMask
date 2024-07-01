import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

# Verificar si la GPU está disponible y establecer el dispositivo
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Definimos las dos redes convolucionales
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 5 * 5, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 64 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Inicializamos las dos redes y las movemos a la GPU si está disponible
net = Net().to(device)
mask = Mask().to(device)

# Definimos el transform para los datos de MNIST
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

# Cargamos el dataset de MNIST
train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST('./data', train=False, download=True, transform=transform)

# Definimos los DataLoaders para los conjuntos de entrenamiento y prueba
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# Definimos la función de pérdida y los optimizadores
criterion = nn.CrossEntropyLoss()
optimizer_net = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)
optimizer_mask = optim.SGD(mask.parameters(), lr=0.01, momentum=0.9)

def print_parameters(layer_name, params):
    print(f"--- {layer_name} ---")
    print(params)

def apply_mask(net, mask):
    # Aplica la máscara a la red: selecciona el 30% de los pesos más altos y desactiva el resto
    with torch.no_grad():
        for net_name, net_param, mask_name, mask_param in zip(net.state_dict(), net.parameters(), mask.state_dict(), mask.parameters()):
            print_parameters("Mask before applying", mask_param.data)
            print_parameters("Net before applying mask", net_param.data)
            
            mask_data = mask_param.data.abs()
            threshold = torch.quantile(mask_data, 0.7)
            mask_applied = (mask_data >= threshold).float()
            
            print_parameters("Mask applied (binary)", mask_applied)
            net_param.data[mask_applied == 0] = 0
            
            print_parameters("Net after applying mask", net_param.data)
            print("\n")

# Listas para almacenar las pérdidas de entrenamiento y las precisiones de validación
train_losses = []
val_accuracies = []
epochs = 10
accuracy_threshold = 0.6

for epoch in range(epochs):
    # Entrenamiento de la red
    net.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_net.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_net.step()
        
        running_loss += loss.item()
        
    
    train_losses.append(running_loss / len(train_loader))
    
    # Retropropagación del error en la máscara
    mask.train()
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer_mask.zero_grad()
        outputs = mask(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_mask.step()
    
    # Aplicamos la máscara a la red
    apply_mask(net, mask)
    
    # Validación de la red
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    val_accuracies.append(accuracy)
    
    print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Validation Accuracy: {accuracy}')
    
    # Paramos el entrenamiento si la precisión en validación supera el 60%
    if accuracy > accuracy_threshold:
        break

# Graficamos la evolución del error y la precisión en validación
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over Epochs')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(val_accuracies, label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Validation Accuracy over Epochs')
plt.legend()

plt.show()

# Evaluamos la red en el conjunto de prueba
net.eval()
correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = net(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = correct / total
print(f'Test Accuracy: {test_accuracy}')
