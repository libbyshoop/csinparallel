using std::string;

class mapper
{
public:
	void map(string key, string value)
	{
		wmr::emit(key, value);
	}
};
