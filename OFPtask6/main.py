#task6

import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
# import matplotlib
# matplotlib.use('TkAgg')

def y_func(y = None, t = None):
    return -y


def br_ln (func, t_grid, y_0):
    y_grid = []
    y_grid.append(y_0)
    for i in range(len(t_grid)-1):
        y_grid.append(y_grid[i] + func(y_grid[i], t_grid[i])*(t_grid[1] - t_grid[0]))
    return y_grid


def br_ln_cor (func, t_grid, y_0):
    y_grid = []
    y_grid.append(y_0)
    h = t_grid[1]-t_grid[0]
    for i in range(len(t_grid)-1):
        y_grid.append(y_grid[i]+(1/2)*h*(func(y_grid[i], t_grid[i])+func(y_grid[i] + h*func(y_grid[i], t_grid[i]), t_grid[i+1])))
    return y_grid

def runge2 (func, t_grid, y_0):
    y_grid = []
    y_grid.append(y_0)
    h = (t_grid[1] - t_grid[0])
    a = 1/4
    for i in range(len(t_grid) - 1):
        k1 = func(y_grid[i], t_grid[i])
        k2 = func(y_grid[i] + k1*h/(2*a), t_grid[i] + h/(2*a))
        y_grid.append(y_grid[i] + h * ((1 - a) * k1 + a * k2))
    return y_grid


def runge4(func, t_grid, y_0):
    y_grid = []
    y_grid.append(y_0)
    h = (t_grid[1] - t_grid[0])

    for i in range(len(t_grid) - 1):
        k1 = func(y_grid[i], t_grid[i])
        k2 = func(y_grid[i] + h*k1/2, t_grid[i] + h/2)
        k3 = func(y_grid[i] + h * k2/2, t_grid[i] + h/2)
        k4 = func(y_grid[i] + h * k3, t_grid[i] + h)
        y_grid.append(y_grid[i] + h/6 * (k1 + 2 * k2 + 2 * k3 + k4))
    return y_grid


def calculate_mul(method, name, true_func, N_x):
    fig, axes = plt.subplots(3, 1)
    fig.suptitle(name + 'method', fontsize=16)

    x_arr = np.linspace(0, 3, N_x)
    y_arr = np.vstack(method(y_func, x_arr, np.linspace(0, 1, 20)))
    axes[0].plot(x_arr,  y_arr)
    axes[1].plot(x_arr, true_func(x_arr), label='real')
    axes[1].plot(x_arr, y_arr[:, 19], label=name)
    axes[2].plot(x_arr, - y_arr[:, 19] + true_func(x_arr), label=name + ' error')
    for ax in axes.reshape(-1):
        ax.legend()

    plt.show()
    return x_arr, - y_arr[:, 19] + true_func(x_arr)


def descrepancy_order(method, true_func, N_min, N_max):
    i = 0
    N = [2*i+1 for i in range(ceil(N_min/2), floor(N_max/2))]
    error_arr = np.zeros(len(N))
    dx_arr = np.zeros(len(N))

    for N_x in N:
        x_arr = np.linspace(0, 10, N_x)
        dx_arr[i] = x_arr[1] - x_arr[0]
        y_arr = method(y_func, x_arr, 1)
        error_curr = np.vstack(- true_func(x_arr)) + np.vstack(y_arr)
        plt.plot(x_arr, error_curr)
        error_arr[i] = error_curr[ceil(N_x/2)-1]
        i += 1
    plt.show()
    return dx_arr, error_arr


# N = 100
# x_arr, error1 = calculate_mul(br_ln, 'br_ln', lambda x: np.exp(-x), N)
#
# x_arr, error2 = calculate_mul(runge2, 'runge2', lambda x: np.exp(-x), N)
#
# x_arr, error3 = calculate_mul(runge4, 'runge4', lambda x: np.exp(-x), N)
#
# plt.plot(x_arr, error1, label='br_ln')
# plt.plot(x_arr, error2, label='runge2')
# plt.plot(x_arr, error3, label='runge4')
# plt.legend()
# plt.show()

N_from, N_to = 100, 1000
dx_arr, descrepancy_arr1 = descrepancy_order(runge2, lambda x: np.exp(-x), N_from, N_to)
dx_arr, descrepancy_arr2 = descrepancy_order(runge4, lambda x: np.exp(-x), N_from, N_to)

fig, ax = plt.subplots(2)
ax[0].plot(dx_arr, descrepancy_arr1, label='runge2')
ax[0].plot(dx_arr, descrepancy_arr2, label='runge4')

print(np.log(descrepancy_arr1))

ax[1].plot(np.log(dx_arr), np.log(descrepancy_arr2) - np.log(descrepancy_arr2)[-1], label='runge4')
ax[1].plot(np.log(dx_arr), np.log(descrepancy_arr1) - np.log(descrepancy_arr1)[-1], label='runge2')

print((np.log(descrepancy_arr2)[-1] - np.log(descrepancy_arr2)[-0])/(np.log(dx_arr)[-1]-np.log(dx_arr)[0]))
print((np.log(descrepancy_arr1)[-1] - np.log(descrepancy_arr1)[-0])/(np.log(dx_arr)[-1]-np.log(dx_arr)[0]))


ax[0].legend()
ax[1].legend()
plt.show()
