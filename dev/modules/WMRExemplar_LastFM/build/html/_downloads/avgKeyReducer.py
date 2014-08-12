def reducer(key, values):
    keys = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']
    keySig, mode = eval(key)
    keySig = keys[int(keySig)]
    if mode == '0':
        keySig += 'm'
    count = 0.0
    for value in values:
        count += float(value)
    Wmr.emit(keySig, count)
