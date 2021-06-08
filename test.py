import numpy as np
a = [1,2,3]
b = a[:]
b[0],b[1],b[2] = 4,5,6
print(a)