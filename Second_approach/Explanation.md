# Second approach notes
Inicializamos una red aleatoria y un vector de varianzas con el mismo numero de pesos.

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
- Definimos otra red y copiamos los pesos en una lista para tener nuestras varianzas
- Sacamoos el dataset y lo separamos en train y test

- Bucle:
    - if epoch 1:
        -Hacemos pruning en la red (temporal) y nos quedamos el 30% por capa.
        - Pasamos las imagenes por la red y sacamos el error. Lo guardamos en una lista
    - else
        - Guardo los pesos de la red y en la temp sumo las varianzas para cambiar los pesos
        - Hago pruning en la temporal, propago, saco error y lo guardo en una lista
        - Si +=mejoras que empeoramientos (lejos de solución):
            - Lista varianzas * (1/0.82)
        - Else (cerca sol):
            - Lista varianzas * (0.82)

# Nueva reunión
- No se suman las varianzas, hay que hacer una normal(0, var) ya que da un numero distinto cada vez y eso nos permite hacer 10 épocas con cambios ya que ese valor se suma a la red.
- No se calcula la tasa de empeoramientos -> solo mejoras para las últimas 10 iteraciones:
    - Si 2/10 de mejoras exactas -> var * 1
    - Si < 1/5 de mejoras (muy cerca de la solución) -> var * 0.82
    - Si > 1/5 de mejoras (muy lejos por lo que hay que hacer más grandes las varianzas) -> var *1/0.82
