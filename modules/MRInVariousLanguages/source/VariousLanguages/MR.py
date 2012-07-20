def mapper(key, value):
    words=key.split()
    for word in words:
        Wmr.emit(word, '1')

def reducer(key, iter):
    sum = 0
    for s in iter:
        sum = sum + int(s)
    Wmr.emit(key, str(sum))
