# Second approach notes
Inicializamos una red aleatoria, un individuo y un vector de varianzas con el mismo numero de pesos.

En las 9 primeras épocas:
- Hacemos una máscara (pruning 70%) con el individuo sumandole la normal(0,varianza)
- Aplicamos la mascara a la red.
- Entrenamos a la red 10 epochs. Guardamos error (media o los 10).
- Pasamos el conjunto de test para saber el accuracy. Guardamos en lista.

En las siguientes igual hasta después del test en el que tenemos que comprobar una cosa (1+1):
- Comprobamos los últimos 10 valores de la lista:
    - Si mejora = 1/5 de las veces multiplicamos var * 1. (Lo dejamos igual)
    - Si mejora < 1/5 de las veces estamos muy cerca de la solución por lo que disminuimos las varianzas multiplicando: var * 0.82.
    - Si mejora > 1/5 de las veces estamos lejos de la solución por lo que aumentamos las varianzas multiplicando: var * (1/0.82).


# Lógica programación
- Definimos red
- Definimos un primer individuo y guardamo sus pesos.
- Definimos otra red de varianzas.
- Sacamos el dataset y lo separamos en train y test

- Normalizamos los valores de las var. (no hace falta)

- epoch = 1

- Bucle:
    - if epoch == 1: (primer individuo)
        - Hacemos una máscara con el individuo (70%).
        - Hacemos pruning en la red.
        - Entrenamos la red 10 epocas
        - Calculamos loss y acc y guardamos en listas.
        - nuevo_ind = pesos(ind)+normal(0,var)
        - epoch+=1

    - else: (resto de individuos)
        - Hacemos una máscara con el nuevo individuo (70%).
        - Hacemos pruning en la red.
        - Entrenamos la red 10 epocas
        - Calculamos loss y acc y guardamos en listas.
        - If loss[-1] < loss(ind):  (Verificamos si el nuevo ind da mejores resultados)
            - ind_dist.load(nuevo_ind.disct)
        - epoch += 1
        - if epoch >=10:
            - Comparamos cuantas mejoras hay en las 10 últimos valores de la lista
            - Si en las últimas 10 losses hay < 1/5 de mejoras:
                - Varianzas * 0.82
            - elif hay = 1/5 mejoras:
                - Varianzas * 1
            - Else:
                - varianzas * (1/0.82)
        - nuevo_ind = pesos(ind)+normal(0,var)

# Nueva reunión
Me he dado cuenta que lo estoy haciendo mal. 
- La red es siempre la misma.
- Genero individuos con los que creo las máscaras (prune 70%)
- entreno la red 10 epocas, guard loss y accuracy. 
- genero otro individuo o me quedo el mismo dependiendo de los resultados
- HAgo 1+1 (tengo que preguntar si sobre la media de los 10 ultimo sindividuos o como)

# Resultados (anterior)
Aunque estamos haciendo lo que hemos pensado, la red que generamos se podría considerar como dummy, ya que está en torno al 10% de acierto con un dataset de 10 clases distintas. Alguna vez consigue en torno al 20%, pero esto no lo podemos considerar como un lottery ticket, aunque si entrenásemos estos modelos con prining, seguramente conseguiríamos muy buenos resultados.

Los experimentos con cnn y pm con más capas no han surtido efecto aunque se hagan muchas épocas


# Nueva reunión 
## Forma de individuos.

Las varianzas hay que inicializarlas de manera que los valores sean grandes. La idea es que el valor de las varianzas vaya bajando a medida que el modelo va "entrenando".

## Forma de vectores binarios

Vamos a generar "individos" en vez de con varianzas y cosas así, con vectores binarios de 0 y 1. 
Tenemos que generar un vectro con un 30% de 1's que después de cada iteración, lo cambiamos cambiando el 5% de 1's a 0's y viceversa de forma aleatoria.

Nos quedamos con el mejor individuo y variamos a partir del mejor ind.

## Comparación

Tenemos que comprobar que estrategia es mejor:

Si esta estrategia (vect) me da un % tengo que intentar la forma anterior para ver si mejora la nueva. ENTRENANDO Y NO ENTRENANDO CON LOS DOS DATASETS.