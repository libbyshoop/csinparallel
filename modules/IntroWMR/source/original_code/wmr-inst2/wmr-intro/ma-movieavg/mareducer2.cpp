#include <string>

class reducer
{
public:
	void reduce(std::string key, wmr::datastream stm)
	{
		using std::string;
		
		string buffer;
		unsigned int count = 0;
		while (!stm.eof())
		{
			stm >> buffer;
			++count;
		}
		
		wmr::emit(key, count);
	}
};
