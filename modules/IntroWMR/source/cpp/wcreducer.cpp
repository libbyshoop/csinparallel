using std::string;
using std::cout;

class reducer
{
public:
    void reduce(std::string key, wmr::datastream stm)
    {
        long sum = 0;
        string value;

        // grab data from the stream until there are no more values
        while (!stm.eof())
        {
            stm >> value;

            // convert the value to a number, then add it to the
            // running total
            sum += wmr::utility::fromString<long>(value);
        }

        // emit the key with its total count
        wmr::emit(key, sum);
    }
};

