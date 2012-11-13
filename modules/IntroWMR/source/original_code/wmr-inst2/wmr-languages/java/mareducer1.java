class Reducer
{
	public void reduce(String key, WmrIterator iter)
	{
		int total = 0;
		int count = 0;
		
		for (String value : iter)
		{
			total += Integer.parseInt(value);
			count++;
		}
		
		double average = (double)total / (double)count;
		Wmr.emit(key, Double.valueOf(average).toString());
	}
}
