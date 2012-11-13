def reducer(key, iter):
    sum = 0
    for s in iter:
        sum = sum + int(s)
    Wmr.emit(key, str(sum))

