/* Mapper for word count */

class Mapper {
  public void mapper(String key, String value) {
    String words[] = key.split(" ");
    int i = 0;
    for (i = 0;  i < words.length;  i++)
      Wmr.emit(words[i], "1");
  }

}

/* Reducer for word count */

class Reducer {
  public void reducer(String key, WmrIterator iter) {
    int sum = 0;
    while (iter.hasNext()) {
      sum += Integer.parseInt(iter.next());
    }
    Wmr.emit(key, Integer.valueOf(sum).toString());
  }

}


