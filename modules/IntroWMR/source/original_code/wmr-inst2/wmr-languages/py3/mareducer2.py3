def reducer(key, iterator):
	count = 0
	
	# Count the number of movies in each bin
	for movie in iterator:
		count = count + 1
	
	Wmr.emit(key, str(count))
