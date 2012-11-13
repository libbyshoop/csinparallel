def reducer(key, iter):
    lst = ""
    for s in iter:
        if (lst == ""):
            lst = s
        else:
            lst = lst + ", " + s
    Wmr.emit(key, lst)

