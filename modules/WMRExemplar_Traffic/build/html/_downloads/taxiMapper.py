def mapper(key, value):
    data = key.split(',')
    if len(data) == 21:
        if data[2] in ('8', '108'):
            Wmr.emit(data[0], "taxi")
    elif len(data) == 32:
        Wmr.emit(data[0], data[10])
