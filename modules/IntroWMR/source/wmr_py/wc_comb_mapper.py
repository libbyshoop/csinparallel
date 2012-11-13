import string

def mapper(key, value):
    counts = dict()
    words=key.split()
    for word in words:
        word = word.strip(string.punctuation)
        word = word.lower()
        if word not in counts:
            counts[word] = 1
        else:
            counts[word] += 1

    for foundword in counts:
        Wmr.emit(foundword, counts[foundword])

