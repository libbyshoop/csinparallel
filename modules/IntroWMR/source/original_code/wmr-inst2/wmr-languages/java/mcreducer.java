/* Reducer for counting total number of each 
 * rating of Netflix data 
 * key is the rating and we simply are counting them
 */

class Reducer {
  public void reduce(String key, WmrIterator iter) {
    int sum = 0;
    String val = "";
    while (iter.hasNext()) {
      val = iter.next();
      if ( val.length() > 0)
	  sum += Integer.parseInt(val);
    }
    Wmr.emit(key, Integer.valueOf(sum).toString());
  }

}
