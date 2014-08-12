def mapper(key, value):
    data = value.split('\t')
    if len(data) == 25:
        keySig = (data[23], data[21])
        confidence = float(data[24]) * float(data[22])
        Wmr.emit(keySig, confidence)
