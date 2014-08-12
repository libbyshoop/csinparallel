def mapper(key, value):
    friends = value.split(',')
    Wmr.emit('Avg:', len(friends))
