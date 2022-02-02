#task 4

import matplotlib.pyplot as plt
import numpy as np

import matplotlib
matplotlib.use('Qt5Agg')

def Bessel_func(x, m):
        return lambda t: np.cos(m * t - x * np.sin(t)) / np.pi


def diff_y(func, x, dx):
    return (func(x + dx) - func(x)) / dx


def Simpson(N, func, a, b):
    x = np.linspace(a, b, N + 1)
    I = 0
    for i in range(N):
        I += (func(x[i]) + 4 * func((x[i] + x[i + 1]) / 2) + func(x[i + 1])) * ((x[i + 1] - x[i]) / 6)

    return I


def trapezoid(y_arr, x_arr):
    I = 0
    dx = x_arr[1] - x_arr[0]
    for i in range(len(x_arr)-1):
        I += (y_arr[i] + y_arr[i+1]) * (dx / 2)

    return I


def Bessel_eq(N, N1, a, b):

    x_arr = np.linspace(a, b, 2*N+1)
    dx = (b - a)/(2 * N)
    print('dx = ', dx)


    J1 = Simpson(N1, Bessel_func(x_arr + dx/2, 1), 0, np.pi)
    J0_1 = Simpson(N1, Bessel_func(x_arr, 0), 0, np.pi)
    J0_2 = Simpson(N1, Bessel_func(x_arr + dx, 0), 0, np.pi)
    diff_J0 = (J0_2 - J0_1) / dx

    fig, ax = plt.subplots(2, 1)

    ax[0].plot(x_arr, J0_1, label = 'J0')
    ax[0].plot(x_arr, J1, label = 'J1')
    ax[0].plot(x_arr, - diff_J0, label = ' J0_diff')
    ax[1].plot(x_arr, diff_J0 + J1, label='J0_diff - J1')

    ax[0].legend()
    ax[1].legend()
    plt.show()

    return np.vstack((x_arr, diff_J0))

def check_accuracy(N_from, N_to, step, N1, a, b):
    for N in range(N_from, N_to+1, step):
        temp = Bessel_eq(N, N1, a, b)
        result = trapezoid(temp[1], temp[0])
        plt.scatter(N, result)
    plt.show()


#check_accuracy(1*10**3,15*10**3, 100, 150, 0, 2*np.pi)
Bessel_eq(15*10**3, 100, 0, 2*np.pi)



