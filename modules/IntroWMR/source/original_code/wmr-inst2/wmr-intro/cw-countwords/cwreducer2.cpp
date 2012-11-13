using std::string;

class reducer
{
public:
    void reduce(string key, wmr::datastream stm)
    {
		string value;
		while (!stm.eof())
		{
			stm >> value;
			wmr::emit(key, value);
		}
	}
};