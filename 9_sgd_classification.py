import numpy as np
import matplotlib.pyplot as plt



def loss_sigmoid(w, x, y):
    M = np.dot(w, x) * y
    return 2 / (1 + np.exp(M))


def df(w, x, y):
    M = np.dot(w, x) * y

    return -2 * (1 + np.exp(M)) ** -2 * np.exp(M) * x * y



x_train = [[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]]
x_train = [x + [1] for x in x_train]
x_train = np.array(x_train)
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

n_train = len(y_train)
w = [0, 0, 0] # начальное значение вектора w
L = 0.005 # шаг изменения веса, сходимости

lm = 0.01 # скорость забывания
N = 50 # число итераций

Q = np.mean([loss_sigmoid(w, x, y) for x, y in zip(x_train, y_train)]) # начальное значение функции потерь
Q_plot = [Q]

for i in range(N):

    k = np.random.randint(len(x_train)) # с 0 до len(x_train)-1
    x = x_train[k]
    y = y_train[k]
    ek = loss_sigmoid(w, x, y)

    w = w - L * df(w, x, y)


    Q = lm * ek + (1 - lm) * Q
    print("Iteration:", i, " Loss:", Q)

    Q_plot.append(Q)

    #Batch gradient descent
    # ks = np.random.randint(0, len(x_train), size = 5)
    # xs = x_train[ks]
    # ys = y_train[ks]
    #
    # ek_mean = np.mean([loss_sigmoid(w, x, y) for x, y in zip(xs, ys)])
    # w_mean = np.mean([df(w, x, y) for x, y in zip(xs, ys)], axis=0) # усредненный градиент по мини-батчу
    #
    # w = w - L * w_mean
    # Q= lm * ek_mean + (1 - lm) * Q
    # print("Iteration:", i, " Loss:", Q)
    # Q_plot.append(Q)

print(w)



line_x = list(range(max(x_train[:, 0])))  # формирование графика разделяющей линии
line_y = [-x * w[0] / w[1] - w[2] / w[1] for x in line_x]


x_0 = x_train[y_train == 1]  # формирование точек для 1-го
x_1 = x_train[y_train == -1]  # и 2-го классов

plt.scatter(x_0[:, 0], x_0[:, 1], color='red')
plt.scatter(x_1[:, 0], x_1[:, 1], color='blue')
plt.plot(line_x, line_y, color='green')

plt.xlim([0, 45])
plt.ylim([0, 75])
plt.ylabel("длина")
plt.xlabel("ширина")
plt.grid(True)
plt.show()
