/* by Scott Boros */
import java.util.*;

class Mapper
{
    public void map(String key, String value)
    {
        HashMap<String, String> mp = new HashMap<String, String>();
        String words[] = key.split(" ");
        String thisword;
        for (int i = 0; i < words.length; i++)
        {
            thisword = (words[i].toLowerCase()).replaceAll("[^A-Za-z]", "");
            Integer screwJava;
            String javaSucks = (String)mp.get(thisword);
            if (javaSucks == null)
            screwJava = new Integer(1);
            else screwJava = new Integer(Integer.parseInt(javaSucks) + 1);
            mp.put(thisword, screwJava.toString());
        }
      Iterator it = mp.entrySet().iterator();
      while (it.hasNext()) {
          Map.Entry pairs = (Map.Entry)it.next();
          Wmr.emit((String)pairs.getKey(), (String)pairs.getValue());
      }

    }
