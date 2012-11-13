class Mapper {
  public void map(String key, String value) {
     Wmr.emit(key, value);
  }
}