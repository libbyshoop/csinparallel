/* by Bjorn Mellem */
#include <iostream>
#include <string>
#include <algorithm>
#include <map>

using std::string;
using std::vector;
using std::cout;
using std::endl;

class mapper
{
  //std::map<std::string, int> combined;
public:
  void map(std::string key, std::string value)
  {
    // split the line into a vector of words using multiple keys
    std::string delim=(".,!? :;\"(){}[]*-/_");
    vector<string> words = wmr::utility::split_multi(key, delim);
    std::map<std::string, int> combined;

    // loop through the vector, and emit each word with a 1
    for (size_t i = 0; i < words.size(); ++i)
      {
	std::transform(words.at(i).begin(), words.at(i).end(), words.at(i).begin(), tolower);
	//wmr::emit<std::string, int>(words.at(i), 1);
	combined[words.at(i)]++;
      }

    std::map<std::string, int>::const_iterator end=combined.end();
    for (std::map<std::string, int>::const_iterator it = combined.begin(); it!=end; ++it)
    wmr::emit<std::string, int>(it->first, it->second);
  }
};
