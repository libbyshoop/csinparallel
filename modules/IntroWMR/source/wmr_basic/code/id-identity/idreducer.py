def reducer(key, iter):
    for s in iter:
        Wmr.emit(key, s)

