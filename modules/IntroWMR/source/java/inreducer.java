class Reducer 
{
    public void reduce(String key, WmrIterator values)
    {
			String lst = "";
			for(String s : values){
				if(lst == "")
					lst = s;
				else
					lst = lst + ", " + s;
			}
			Wmr.emit(key, lst);
    }
}

