import numpy as np
import matplotlib.pyplot as plt


def loss(w, x, y):
    M = np.dot(w, x) * y
    return 2/ (1 + np.exp(M))



def df(w, x, y):
    L1 = 1.0  # коэффициент регуляризации L1, отбор признаков
    # L2 = 20.0  # коэффициент регуляризации L2, уменьшение переобучения
    M = np.dot(w, x) * y

    return -2 * (1 + np.exp(M)) ** -2 * np.exp(M) * x * y + L1 * np.sign(w)
    # return -2 * (1 + np.exp(M)) ** -2 * np.exp(M) * x * y + 2 * L2 * w

x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = [x + [10*x[0], 10*x[1], 5*(x[0]+x[1])] for x in x_train]
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])


fn = len(x_train[0]) # размер признакового пространства
n_train = len(x_train)  # размер обучающей выборки

print(fn)
print(n_train)

w = np.zeros(fn)        # начальные весовые коэффициенты
nt = 0.00001             # шаг сходимости SGD
lm = 0.01               # скорость "забывания" для Q
N = 5000                 # число итераций SGD

Q = np.mean([loss(w, x, y) for x, y in zip(x_train, y_train)])  # начальное значение функции потерь
Q_plot = [Q]

for i in range(N):
    k = np.random.randint(0, n_train - 1)
    ek = loss(w, x_train[k], y_train[k])
    w = w - nt * df(w, x_train[k], y_train[k])
    Q = lm * ek + (1 - lm) * Q
    print("Iteration:", i, " Loss:", Q)
    Q_plot.append(Q)


Q = np.mean([loss(x, w, y) for x, y in zip(x_train, y_train)]) # истинное значение эмпирического риска после обучения
print(w) # Выделил линейно независимые признаки, а все остальные отбросил
print(Q)

plt.plot(Q_plot)
plt.grid(True)
plt.show()
