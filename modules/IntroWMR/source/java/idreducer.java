class Reducer {
  public void reduce(String key, WmrIterator values) {
    for (String value : values) {
      Wmr.emit(key, value);
    }
  }
}