def reducer(key, values):
    oneHops = set()
    twoHops = {}
    for value in values:
        node, hops = eval(value)
        oneHops.add(node)
        hops = hops.split(',')
        for hop in hops:
            if hop in twoHops:
                twoHops[hop] += 1
            else:
                twoHops[hop] = 1
    n = len(oneHops)
    if n < 2:
        Wmr.emit(key, 0)
    else:
        total = 0.0
        for hop in oneHops:
            if hop in twoHops:
                total += twoHops[hop]
        cc = total / (n * (n-1))
        Wmr.emit(key, cc)
