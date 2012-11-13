def reducer(key, iterator):
	total = 0	# The overall rating of the movie
	count = 0	# The number of ratings the movie had
	
	# Find the total rating and number of ratings for this movie
	for rating in iterator:
		total = total + int(rating)
		count = count + 1
	
	# Calculate the average rating of this movie
	average = float(total / count)
	Wmr.emit(key, str(average))
