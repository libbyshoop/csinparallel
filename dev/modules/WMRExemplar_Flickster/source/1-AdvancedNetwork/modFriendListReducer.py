def reducer(key, values):
    neighbors = set()
    for value in values:
        if len (neighbors) > 50:
            Wmr.emit(key, ','.join(neighbors))
            neighbors.clear() 
        neighbors.add(value)
    if len(neighbors) > 0:
        Wmr.emit(key, ','.join(neighbors))
