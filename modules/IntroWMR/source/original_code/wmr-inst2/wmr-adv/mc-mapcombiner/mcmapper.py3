# by Boyang Wei
def mapper(key, value):
  words = key.split()
  assoArr = {}
  for word in words:
    assoArr[word] = 0
  for word in words:
    assoArr[word] = assoArr[word]+1
  for term in assoArr:
    Wmr.emit(term, assoArr[term])
