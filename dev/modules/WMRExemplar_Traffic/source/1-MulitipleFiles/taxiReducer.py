def reducer(key, values):
    isTaxi = False
    dayOfWeek = ''
    for value in values:
        if value == 'taxi':
            isTaxi = True
        else:
            dayOfWeek = value
    Wmr.emit(dayOfWeek, 1)
