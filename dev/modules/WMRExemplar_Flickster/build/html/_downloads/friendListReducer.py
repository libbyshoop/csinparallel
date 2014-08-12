def reducer(key, values):
    neighbors = set()
    for value in values:
        neighbors.add(value)
    output = ','.join(neighbors)
    Wmr.emit(key, output)
