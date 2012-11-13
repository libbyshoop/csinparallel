using std::string;
using std::vector;
using std::map

class mapper
{
public:
    void map(std::string key, std::string value)
    {
    	//typedef std::tr1::unordered_map< std::string, int > hashmap;
    	map<string, int> counts;

        // split the line into a vector of words
        vector<string> words = wmr::utility::split(key, ' ');

        // loop through the vector, and emit each word with a 1
        for (size_t i = 0; i < words.size(); ++i) {
        	int current_count = counts[words.at(i)];
        	if (current_count) {
        		counts[words.at(i)] = current_count + 1;
        	} else {
        		counts[words.at(i)] = 1;
        	}
        }

        // go through map and emit counts for each word
        for(map<string, int>::iterator i = counts.begin(), e = counts.end() ; i != e ; ++i ) {

            wmr::emit(i->first, i->second);
        }
    }
};

