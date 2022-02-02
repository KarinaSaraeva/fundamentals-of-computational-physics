import numpy as np
import matplotlib.pyplot as plt


def banded_matrix(a, b, c, h, n):
    matrix = np.zeros((n, n))

    for i in range(n - 1):
        matrix[i][i] = b/(h**2)
        matrix[i][i + 1] = c/(h**2)
        matrix[i + 1][i] = a/(h**2)

    matrix[0][0] = 1
    matrix[0][1] = 0
    matrix[n - 1][n - 2] = 0
    matrix[n - 1][n - 1] = 1

    return matrix


def gauss_for_banded(matrix, d):
    buffer = np.copy(matrix)
    for i in range(len(buffer[0]) - 1):
        temp = buffer[i + 1, i] / buffer[i, i]
        buffer[i + 1, :] = buffer[i + 1, :] - buffer[i, :] * temp
        d[i + 1] = d[i + 1] - d[i] * temp
    return buffer, d


def x_for_banded(matrix, d):
    n = len(d)
    x = np.zeros(n)

    x[n - 1] = d[n - 1]/matrix[n - 1][n - 1]
    for i in range(n - 1):
        x[n - 2 - i] = (1/matrix[n - 2 - i][n - 2 - i])*(d[n - 2 - i] - matrix[n - 2 - i][n - i - 1]*(x[n - 1 - i]))
    return x


def sweep_2nd_deriv(func, a, b, n, y_a = None, y_b = None, der_y_a = None, der_y_b = None):

    x = np.linspace(a, b, n)
    h = x[1]-x[0]
    matrix = banded_matrix(1, -2, 1, h, n)

    right_part = func(x)
    right_part[0] = y_a
    right_part[n-1] = y_b

    triagonal, newright_part = gauss_for_banded(matrix, right_part)

    y = x_for_banded(triagonal, newright_part)

    plt.plot(x, y)
    plt.show()

    return 0


sweep_2nd_deriv(lambda x: np.sin(2*x), 0, np.pi, 50, y_a = 0.6, y_b = 0.8)




