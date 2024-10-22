# Second approach notes
Inicializamos una red aleatoria y un vector con el mismo numero de pesos pero con numeros grandes
Podo la red un 70%, propago la red y saco el error.

Con el error aplicamos la estrategia evolutiva 1+1 para calcular las nuevas varianzas.
Aplicamos las varianzas para tener nuevos pesos.

Loss solo se usa para calcular las nuevas varianzas.

## Estrategia evolutiva 1+1

Vector con 10 unidades y 10 varianzas (asociadas a los pesos).
- Obtenemos un nuevo vector (hacemos una normal 0 y la varianza).

Hay que conseguir que converja a varianzas pequeñas (último resultado cerca de 0) -> Cambio de varianzas regla 1/5.
- Guardo
- Si más de 1/5 de las veces mejora -> lejos de la solución. Si mejora mucho en las últimas veces lejos de la sol.
Si eso pasa, bajamos varianzas.

Cambiamos varianzas por 0,82 o por 1/0,82


PROBLEMA: No se puede hacer una sola vez(varios experimentos para ver si tiene sentido con distintas inicializaciones)


## Me lo vuelven a explicar

Para cambiar las 

Guardamos el loss en una lista y comparamos las 10 ultimas veces. Si 1/5 de las 10 ultimas veces, mejora o hay muchas mejoras -> estamos lejos de la solución -> aumentamos sigma x(1/0.82)

Si hay muchas peoras -> Estamos cerca de la solución -> disminuimos sigma x(0.82)


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

# Poblemas a preguntár
En el approach inicial habían dicho que si 1/5 de las veces mejoraba -> *1/0,82. Pero en esa idea puede mejorar y empeorar a la vez un numero de veces. 

Qué pasaría si los ultimos 10 errores son 5 y 5 en cuanto a mejora y peora y se hace un bucle al mejorar y empeorar siempre. Estaríamos multilicando siempre por lo mismo y llegando siempre a los mismos pesos

# Nueva reunión
- No se suman las varianzas, hay que hacer una normal(0, var) ya que da un numero distinto cada vez y eso nos permite hacer 10 épocas con cambios ya que ese valor se suma a la red.
- No se calcula la tasa de empeoramientos -> solo mejoras para las últimas 10 iteraciones:
    - Si 2/10 de mejoras exactas -> var * 1
    - Si < 1/5 de mejoras (muy cerca de la solución) -> var * 0.82
    - Si > 1/5 de mejoras (muy lejos por lo que hay que hacer más grandes las varianzas) -> var *1/0.82
