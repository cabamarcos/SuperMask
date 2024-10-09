def improvements(losses):
    mejoras = 0
    peoras = 0

    # Determinar cuántos valores comparar
    num_valores = min(10, len(losses))

    for i in range(-num_valores, -1):
        if losses[i] >= losses[i+1]:
            mejoras += 1
        elif losses[i] < losses[i+1]:
            peoras += 1

    if mejoras >= peoras:
        return 0
    else:
        return 1
