import java.util.HashMap;
import java.util.Map;

class Mapper
{
    public void map(String key, String value)
    {
    	Map<String, Integer> counts = new HashMap<String, Integer>();
    	

        String words[] = key.split(" ");
        for (int i = 0; i < words.length; i++)
        {
        	Integer count = counts.get(words[i]);
        	if (count != null) {
        		counts.put(words[i], count + 1);
        	} else {
        		counts.put(words[i], 1);
        	}
            
        }

        for (Map.Entry<String, Integer> entry : counts.entrySet()) {
        	Wmr.emit(entry.getKey(), entry.getValue().toString());
        }
    }
}

