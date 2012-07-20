class Mapper
{
public:
    void mapper(string key, string value)
    {
        char delim = ' ';
        vector splits = Wmr::split(key, delim);
        
        for (unsigned int i = 0; i < splits.size(); ++i)
        {
            Wmr::emit(splits.at(i), "1");
        }
    }
};

class Reducer
{
public:
    void reducer(string key, WmrIterator iter)
    {
        long count = 0;
        while (iter != WmrIterator::end())
        {
            count += Wmr::strToLong(*iter++);
        }
        
        Wmr::emit(key, Wmr::longToStr(count));
    }
};
