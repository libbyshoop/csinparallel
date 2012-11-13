#include <string>
#include <vector>

class mapper
{
public:
	void map(std::string key, std::string value)
	{
		using std::string;
		using std::vector;
		
		vector<string> fields = wmr::utility::split(key, ',');
		string id = fields.at(0);
		string rating = fields.at(2);
		
		wmr::emit(id, rating);
	}
};
