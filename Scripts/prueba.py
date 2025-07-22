import torch

# Crear un diccionario de estado controlado con valores sencillos para la red 1 (pesos)
state_dict_red1 = {
    'layer1.weight': torch.tensor([[1.0, 2.0], [3.0, 4.0]]),  # Pesos sencillos
}

# Crear un diccionario de estado controlado con valores sencillos para la red 2 (varianzas)
state_dict_red2 = {
    'layer1.weight': torch.tensor([[0.5, -0.5], [1.0, -1.0]]),  # Desviaciones estándar (positivas y negativas)
}

# Crear el nuevo diccionario de estado donde sumamos o restamos los pesos
state_dict_suma = {}
for key in state_dict_red1:
    if state_dict_red1[key].size() == state_dict_red2[key].size():  # Asegurar que las dimensiones coincidan
        # Asegurar que las desviaciones estándar sean positivas para generar el ruido
        std_dev = torch.abs(state_dict_red2[key])
        
        # Generamos los valores aleatorios con una distribución normal usando torch.normal
        noise = torch.normal(0, std_dev)  # Media = 0, Desviación estándar = std_dev
        
        # Crear una máscara para determinar si debemos sumar o restar
        mask_negativa = state_dict_red2[key] < 0  # Máscara de valores negativos
        
        # Si la desviación estándar es negativa, restamos; si es positiva, sumamos
        state_dict_suma[key] = state_dict_red1[key] + noise * torch.where(mask_negativa, -1, 1)  
    else:
        # Si los tamaños no coinciden, copiamos directamente
        state_dict_suma[key] = state_dict_red1[key]

# Mostrar resultados para verificar
print("Pesos originales (Red 1):")
print(state_dict_red1['layer1.weight'])

print("\nDesviaciones estándar (Red 2):")
print(state_dict_red2['layer1.weight'])

print("\nRuido generado:")
print(noise)

print("\nMáscara (para saber si se resta):")
print(mask_negativa)

print("\nPesos finales (después de suma/resta):")
print(state_dict_suma['layer1.weight'])
