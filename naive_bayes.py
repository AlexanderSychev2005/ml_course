# Пример реализации наивного гауссовского байесовского классификатора

import numpy as np


x_train = np.array([[10, 50], [20, 30], [25, 30], [20, 60], [15, 70], [40, 40], [30, 45], [20, 45], [40, 30], [7, 35]])
y_train = np.array([-1, 1, 1, -1, -1, 1, 1, -1, 1, -1])

mw1, ml1 = np.mean(x_train[y_train == 1], axis= 0) # средние значения признаков для класса 1
mw2, ml2 = np.mean(x_train[y_train == -1], axis= 0) # средние значения признаков для класса -1

sw1, sl1 = np.var(x_train[y_train == 1], axis=0, ddof=0) # дисперсии признаков для класса 1, ddof=0 - для выборки N+1
sw2, sl2 = np.var(x_train[y_train == -1], axis=0, ddof=0) # дисперсии признаков для класса -1


print(f"Class 1: mean width = {mw1}, mean length = {ml1}, var width = {sw1}, var length = {sl1}")
print(f"Class -1: mean width = {mw2}, mean length = {ml2}, var width = {sw2}, var length = {sl2}")


x = [10, 40]  # ширина, длина жука

a1 = lambda x: -0.5 * np.log(sw1 * sl1) - (x[0] - mw1) ** 2 / (2 * sw1) - (x[1] - ml1) ** 2 / (2 * sl1)
a2 = lambda x: -0.5 * np.log(sw2 * sl2) - (x[0] - mw2) ** 2 / (2 * sw2) - (x[1] - ml2) ** 2 / (2 * sl2)

y = np.argmax([a1(x), a2(x)]) * 2 -1  # выбираем класс с максимальной апостериорной вероятностью

print(f"Predicted class (-1 - гусеница, +1 - божья коровка): {y}")

pr = []
for x in x_train:
    pr.append(np.argmax([a2(x), a1(x)]) * 2 - 1)


pr = np.array(pr)
Q = np.mean(pr != y_train) # доля ошибок
print(Q)