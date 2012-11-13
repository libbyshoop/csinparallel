def mapper(key, value):
    fields=key.split(',')
    Wmr.emit(fields[2], '1')

