#include <string>

class mapper
{
public:
	void map(std::string key, std::string value)
	{
		using std::string;
		
		double average = wmr::utility::fromString<double>(value);
		double bin = 0.0;
		
		if (0 <= average && average < 1.25)
		{
			bin = 1.0;
		}
		else if (1.25 <= average && average < 1.75)
		{
			bin = 1.5;
		}
		else if (1.75 <= average && average < 2.25)
		{
			bin = 2.0;
		}
		else if (2.25 <= average && average < 2.75)
		{
			bin = 2.5;
		}
		else if (2.75 <= average && average < 3.25)
		{
			bin = 3.0;
		}
		else if (3.25 <= average && average < 3.75)
		{
			bin = 3.5;
		}
		else if (3.75 <= average && average < 4.25)
		{
			bin = 4.0;
		}
		else if (4.25 <= average && average < 4.75)
		{
			bin = 4.5;
		}
		else if (4.75 <= average && average <= 5.0)
		{
			bin = 5.0;
		}
		else
		{
			bin = 99.0;
		}
		
		wmr::emit(bin, key);
	}
};
