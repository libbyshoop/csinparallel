def mapper(key, value):
    words=value.split()
    for word in words:
        Wmr.emit(word, key key + ": " + value)

