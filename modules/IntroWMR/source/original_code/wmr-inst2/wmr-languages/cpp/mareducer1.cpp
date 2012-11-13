#include <string>
#include <sstream>
#include <iostream>

class reducer
{
public:
	void reduce(std::string key, wmr::datastream stm)
	{
		using std::string;
		using std::stringstream;
		
		unsigned int total = 0;
		unsigned int count = 0;
		unsigned int temp = 0;
		string buffer;
		
		while (!stm.eof())
		{
			stm >> buffer;
			temp = wmr::utility::fromString<unsigned int>(buffer);
			buffer.clear();
			
			total += temp;
			++count;
		}
		
		double average = (double)total / (double)count;
		wmr::emit(key, average);
	}
};
