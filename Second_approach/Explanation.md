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