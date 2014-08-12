def reducer(key, values):
    count = 0
    total = 0
    for value in values:
        count += 1
        total += int(value)
    Wmr.emit(key, total / count)
