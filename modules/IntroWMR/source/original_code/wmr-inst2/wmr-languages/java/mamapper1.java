class Mapper
{
	public void map(String key, String value)
	{
		String[] fields = key.split("\\,");
		String id = fields[0];
		String rating = fields[2];
		Wmr.emit(id, rating);
	}
}
