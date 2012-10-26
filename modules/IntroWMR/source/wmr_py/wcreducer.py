def reducer(key, values):
  sum = 0
  for count in values:
    sum += int(count)
  Wmr.emit(key, sum)

