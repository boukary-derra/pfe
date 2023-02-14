import numpy as np

tp = np.zeros((3, 3))
print(tp)

w = np.array([1/16, 1/8, 1/16, 1/8, 1/4, 1/8, 1/16, 1/8, 1/16]).reshape(3, 3)

(a, b) = w.shape
for i in range(a):
    for j in range(b):
        print(i, j)
        tp[i, j]=w[i, j]

print(tp)
