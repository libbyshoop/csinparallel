def reducer(key, values):
    isMatch = False
    songPairs = []
    for value in values:
        flag, data = eval(value)
        if flag == "term":
            isMatch = data
        elif flag == "song":
            songPairs.append(data)
    if isMatch:
        for keySig, confidence in songPairs:
            Wmr.emit(keySig, confidence)
