def mapper(key, value):
    data = key.split(',')
    casualties = data[8]
    urbanOrRural = data[29]
    Wmr.emit(urbanOrRural, casualties)
