import numpy as np
import matplotlib.pyplot as plt


def predict_poly(x, koeff):
    res = 0
    xx = [x ** (len(koeff) - n - 1) for n in range(len(koeff))]

    for i, k in enumerate(koeff):
        res += k * xx[i]

    return res




x = np.arange(0, 10.1, 0.1)
y = 1 / (1 + 10 * np.square(x))

X_train, y_train = x[::2], y[::2]
print(X_train)
print(y_train)
X_test, y_test = x[1::2], y[1::2]

N = len(x)
z_train = np.polyfit(X_train, y_train, 54) # коэффициенты полинома степени 3
print(z_train)


y_pred = predict_poly(X_test, z_train)

plt.figure()
plt.scatter(X_test, y_test, color='blue', label='test data')
plt.scatter(X_train, y_train, color='red', label='train data')
plt.plot(X_test, y_pred, color='green', label='predicted')
plt.legend()
plt.show()