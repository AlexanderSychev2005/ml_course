# вычисление собственных векторов и собственных чисел

import numpy as np
import matplotlib.pyplot as plt

SIZE = 1000
np.random.seed(123)
x = np.random.normal(size=SIZE)
y = np.random.normal(size=SIZE)

z = (x + y) / 2

F = np.vstack([x, y, z])
print("F shape:", F.shape)


FF = 1 / SIZE * F @ F.T
L, W = np.linalg.eig(FF) # собственные числа и собственные векторы
print("Eigenvalues:", L)
print("Eigenvectors:\n", W)

WW = sorted(zip(L, W.T), key=lambda lx: lx[0], reverse=True)
WW = np.array([w[1] for w in WW])

print(sorted(L, reverse=True))

