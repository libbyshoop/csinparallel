def reducer(key, values):
    count = 0
    for value in values:
        count += int(value)
    Wmr.emit(key, count)
