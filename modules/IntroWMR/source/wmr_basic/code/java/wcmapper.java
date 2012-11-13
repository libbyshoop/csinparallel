class Mapper
{
    public void map(String key, String value)
    {
        String words[] = key.split(" ");
        for (int i = 0; i < words.length; i++)
        {
            Wmr.emit(words[i], "1");
        }
    }
}

