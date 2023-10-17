import numpy as np
np.random.random((3,10,2))
a = np.random.random((3,4))
b = np.random.random((4, 10))
print(np.matmul(a, b))
print(np.dot(a, b))
print(np.dot(a, b).shape)
print(np.ones(a))
print(np.maximum(a))

