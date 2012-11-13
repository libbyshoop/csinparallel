def mapper(key, value):
	fields = key.split(',')
	movieid = fields[0]
	rating = fields[2]
	Wmr.emit(movieid, rating)
