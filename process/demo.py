def __genBIT():
    _dict = {}
    index = 1
    for bits in range(16):
        x= '{0:08b}'.format(bits)
        U = 0
        for bit in range(7):
            U += 1 if abs(int(x[bit]) - int(x[bit+1])) > 0 else 0
        if U <= 2:
            _dict[x] = index
            index+=1
    return _dict


d = __genBIT()

print(d)