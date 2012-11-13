def mapper(key, value):
	average = float(value)
	bin = 0.0
	
	# Place the movie into a bin based on its average rating
	# Doing this will allow us to fill bins with movie ids
	if (0 <= average and average < 1.25):
		bin = 1.0
	elif (1.25 <= average and average < 1.75):
		bin = 1.5
	elif (1.75 <= average and average < 2.25):
		bin = 2.0
	elif (2.25 <= average and average < 2.75):
		bin = 2.5
	elif (2.75 <= average and average < 3.25):
		bin = 3.0
	elif (3.25 <= average and average < 3.75):
		bin = 3.5
	elif (3.75 <= average and average < 4.25):
		bin = 4.0
	elif (4.25 <= average and average < 4.75):
		bin = 4.5
	elif (4.75 <= average and average <= 5.0):
		bin = 5.0
	else:
		bin = 99.0	# Error, movie does not fit any bin
	
	Wmr.emit(str(bin), key)
