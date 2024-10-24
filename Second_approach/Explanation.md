# Second approach notes
Inicializamos una red aleatoria y un vector de varianzas con el mismo numero de pesos.

- Normalizamos los valores de la red entre 0 y 1 en cada iter
- Normalizamos los valores de las var entre 0,9 y 1.

En las 9 primeras épocas:
- Sumamos a cada peso de la red += normal(0,correspondiente varianza).
- Hacemos pruning del 70% de los pesos más bajos.
- Propagamos las imágenes y cogemos el error. Guardamos en lista.
- Pasamos el conjunto de test para saber el accuracy. Guardamos en lista.

En las siguientes igual hasta después del test en el que tenemos que comprobar una cosa:
- Comprobamos los últimos 10 valores de la lista:
    - Si mejora = 1/5 de las veces multiplicamos var * 1. (Lo dejamos igual)
    - Si mejora < 1/5 de las veces estamos muy cerca de la solución por lo que disminuimos las varianzas multiplicando: var * 0.82.
    - Si mejora > 1/5 de las veces estamos lejos de la solución por lo que aumentamos las varianzas multiplicando: var * (1/0.82).


# Lógica programación
- Definimos red
- Definimos otra red de varianzas.
- Sacamos el dataset y lo separamos en train y test

- Normalizamos los valores de las var entre 0,9 y 1.

- Bucle:
    - if epoch <= 9:
        - Normalizamos los valores de la red entre 0 y 1.
        - Sumamos a cada peso de la red += normal(0,correspondiente varianza)
        - Hacemos pruning en la red (temporal) y nos quedamos el 30% por capa.
        - Pasamos las imagenes por la red, sacamos error y test acc.
        - if loss >= max(loss):
            - net_dist.load(varied_net.disct)
        - else:
            pass
        - Guardamos loss y acc en listas
    - else
        - Normalizamos los valores de la red entre 0 y 1 en cada iter
        - Normalizamos los valores de las var entre 0,9 y 1.
        - Sumamos a cada peso de la red += normal(0,correspondiente varianza)
        - Hacemos pruning en la red (temporal) y nos quedamos el 30% por capa.
        - Pasamos las imagenes por la red, sacamos error y test acc.
        - if loss >= max(loss):
            - net_dist.load(varied_net.disct)
        - else:
            pass
        - Guardamos loss y acc en listas
        - Comparamos cuantas mejoras hay en las 10 últimos valores de la lista
        - Si en las últimas 10 losses hay < 1/5 de mejoras:
            - Varianzas * 0.82
        - elif hay = 1/5 mejoras:
            - Varianzas * 1
        - Else:
            - varianzas * (1/0.82)

# Nueva reunión
Si da un buen resultado en loss (mejor que el mejor loss que haya habido hasta el momento) guardamos la red para que la prox iteración se haga con los pesos nuevos.

Normalizar los datos:
- red de 0 a 1
- varianzas de 0.9 a 1