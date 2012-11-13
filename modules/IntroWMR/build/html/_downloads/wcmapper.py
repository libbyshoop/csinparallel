def mapper(key, value):
    words=key.split()
    for word in words:
        Wmr.emit(word, '1')

