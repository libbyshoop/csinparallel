using std::string;
using std::vector;

class mapper
{
public:
    void map(std::string key, std::string value)
    {
        // split the line into a vector of words
        vector<string> words = wmr::utility::split(key, ' ');

        // loop through the vector, and emit each word with a 1
        for (size_t i = 0; i < words.size(); ++i)
            wmr::emit(words.at(i), 1);
    }
};

