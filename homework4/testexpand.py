# coding = utf-8


import numpy as np

a = [[0, 0, 2], [2, 2, 1], [0, 0, 1]]
a = np.array(a)

x0 = np.expand_dims(a, axis=0)
x1 = np.expand_dims(a, axis=1)
x2 = np.expand_dims(a, axis=2)

m = np.moveaxis(x0, [1,2], [0,1])
# print(x2)
print(np.expand_dims(a, 2))
b = np.expand_dims(a, 2)
c = np.concatenate([b, b], axis=-1)