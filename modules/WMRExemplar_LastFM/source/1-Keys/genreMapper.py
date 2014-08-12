def mapper(key, value):
    genre = "rock"
    data = value.split('\t')
    if key == "metadata" and len(data) == 25:
        artist = data[5]
        keySig = (data[23], data[21])
        confidence = float(data[24]) * float(data[22])
        Wmr.emit(artist, ("song", (keySig, confidence)))
    elif key == "term":
        artist = data[0]
        for triplet in data[1:]:
            term, freq, weight = triplet.split(',')
            if term == genre and float(weight) > 0.5:
                Wmr.emit(artist, ("term", True))
