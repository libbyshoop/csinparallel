class Reducer
{
    public void reduce(String key, WmrIterator values)
    {
        int sum = 0;
        for (String value : values)
        {
            sum += Integer.parseInt(value);
        }
        Wmr.emit(key, Integer.valueOf(sum).toString());
    }
}

