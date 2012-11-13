class Reducer
{
	public void reduce(String key, WmrIterator iter)
	{
		int count = 0;
		for (String value : iter)
		{
			count++;
		}
		
		Wmr.emit(key, Integer.valueOf(count).toString());
	}
}
