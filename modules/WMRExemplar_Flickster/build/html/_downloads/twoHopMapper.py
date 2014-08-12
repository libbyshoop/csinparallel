def mapper(key, value):
    friends = value.split(',')
    for friend in friends:
        Wmr.emit(friend, (key, value))
