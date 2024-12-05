def improvements(lista):
    # Verificar si la lista tiene al menos 10 elementos
    if len(lista) < 10:
        raise ValueError("La lista debe contener al menos 10 elementos.")
    
    # Inicializar la variable mejoras
    mejoras = 0
    
    # Tomar los últimos 10 elementos de la lista
    ultimos_10 = lista[-10:]
    
    # Comparar los elementos uno a uno
    for i in range(len(ultimos_10) - 1):
        if ultimos_10[i + 1] <= ultimos_10[i]:
            mejoras += 1
    
    # Devolver el número de mejoras
    return mejoras