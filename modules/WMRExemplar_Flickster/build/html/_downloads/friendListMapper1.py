def mapper(key, value):
    if not (key in ('', None) or value in ('', None)):
        Wmr.emit(key, value)
        Wmr.emit(value, key)
