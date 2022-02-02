import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
import matplotlib
matplotlib.use('TkAgg')


a_0 = 1
a_1 = 0.002
w_0 = 5.1
w_1 = 25.5
T = 2*np.pi

def y(t):
    return a_0*np.sin(w_0*t) + a_1*np.cos(w_1*t)

def y1(t):
    return np.exp()

def Fourier_matrices(n):
    F_plus = np.zeros((n, n), dtype='complex_')
    F_minus = np.zeros((n, n), dtype='complex_')
    c_plus = 1 / (n ** 0.5)
    c_minus = c_plus
    for j in range(n):
        for k in range(n):
            F_plus[j, k] = c_plus * np.exp(2 * np.pi * (1j) * j * k / n)
            F_minus[j, k] = c_minus * np.exp(- 2 * np.pi * (1j) * j * k / n)

    return F_plus, F_minus

def for_fan():
    for i in range(10, 100, 2):
        Direct, Inverse = Fourier_matrices(i)
        plt.scatter(np.real(Direct), np.imag(Direct))
    plt.show()

def find_w_grid(N, T):
    omega = 2*np.pi*N/T
    w_grid = np.array([j*omega/N if j < N/2 else -omega+j*omega/N for j in range(N)])
    plt.plot(np.linspace(0, len(w_grid), len(w_grid)), w_grid)
    plt.show()
    return w_grid

def FT(T, N, func):
    t_grid = np.linspace(0, T, N)
    y_grid = func(t_grid)

    # y_grid = Hann_window(t_grid, y_grid, N)
    w_grid = find_w_grid(N, T)
    Direct, Inverse = Fourier_matrices(N)
    FT_y_grid = np.dot(Direct, y_grid)

    after_y_grid = np.dot(Inverse, FT_y_grid)

    fig, ax = plt.subplots(3)
    ax[0].plot(t_grid, y_grid)
    ax[2].plot(t_grid, y_grid - after_y_grid)
    abs_FT = abs(FT_y_grid) ** 2

    order_abs_FT = np.zeros(N)
    order_w_grid = np.zeros(N)

    order_abs_FT[:int(N / 2)] = abs_FT[int(N / 2):]
    order_abs_FT[int(N / 2):] = abs_FT[:int(N / 2)]

    order_w_grid[:int(N / 2)] = w_grid[int(N / 2):]
    order_w_grid[int(N / 2):] = w_grid[:int(N / 2)]


    ax[1].plot(order_w_grid, np.log(order_abs_FT))
    test = (w_grid)**2
    plt.show()


def Hann_window(t_grid, y_grid, n):
    def window_func(k):
        return 0.5*(1 - np.cos(2*np.pi*k/(n - 1)))

    plt.plot(t_grid, y_grid)
    hann_y = np.zeros(n)
    for i in range(n):
        hann_y[i] = y_grid[i]*window_func(i)
    plt.plot(t_grid[:n], hann_y)
    plt.show()
    return hann_y

FT(T, 300, y)



