import numpy as np

# Vector de varianzas aleatorias
varianzas = np.random.rand(4)

# Calcula las desviaciones estándar correspondientes
desviaciones_estandar = np.sqrt(varianzas)

# Genera un nuevo vector de valores de acuerdo a una normal N(0, varianza)
nuevas_valores = np.random.normal(0, desviaciones_estandar)

print("Vector de varianzas:", varianzas)
print("Desviaciones estándar:", desviaciones_estandar)
print("Nuevo vector de valores generados:", nuevas_valores)
