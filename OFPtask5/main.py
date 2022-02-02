# task 5

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib
matplotlib.use('TkAgg')


n = 8
x = np.array([1 + k / n for k in range(n + 1)])


def recur(arr):
    if len(arr[-1]) > 1:
        div_dif_arr = np.zeros(len(arr[-1]) - 1)
        for i in range(len(arr[-1]) - 1):
            div_dif_arr[i] = (arr[-1][i] - arr[-1][i+1]) / (arr[0][i]-arr[0][(len(arr[0])-len(arr[-1])+i+1)])  # есть элемент нового массива
        arr.append(div_dif_arr)
        print(arr)
        return recur(arr)
    else:
        return 0


def find_y(arr, z):
    y = 0
    for i in range(len(arr)-1):
        y += arr[i+1][0]*find_prod(i, arr, z)
    return y


def find_prod(k, arr, z):
    product = 1
    for i in range(k): #the last is k-1
        product *= (z - arr[0][i])
    return product


def interpolation(x_grid, func):
    list_div_dif = []
    list_div_dif.append(x)
    list_div_dif.append(func(x))
    recur(list_div_dif)
    y_grid = np.zeros(len(x_grid))
    for i in range(len(x_grid)):
        y_grid[i] = find_y(list_div_dif, x_grid[i])
    fig, ax = plt.subplots(2, 1)
    ax[0].plot(x_grid, y_grid)
    ax[0].plot(x_grid, np.log(x_grid))
    ax[0].scatter(x, np.log(x)) #true
    error_arr = y_grid - np.log(x_grid)
    ax[1].plot(x_grid, error_arr)


    print('points of interpolation =', x)
    print('zeros of error_arr = ', x_grid[error_arr == 0])

    plt.show()
    return x_grid, error_arr

def max_error(x_grid, dif_func):
    dif_arr = np.zeros(len(x_grid))
    omega_arr = np.zeros(len(x_grid))
    list = []
    list.append(x)
    for i in range(len(x_grid)):
        omega_arr[i] = find_prod(len(x), list, x_grid[i])
        dif_arr[i] = dif_func(x_grid[i])

    max_dif_arr = np.zeros(len(x_grid))
    for i in range(len(x) - 1):
        a = x[i]
        b = x[i+1]
        slice = dif_arr[(x_grid <= b)*(x_grid >= a)]
        max_dif_arr[(x_grid <= b)*(x_grid >= a)] = np.amax(slice)

    an_error = omega_arr * max_dif_arr / math.factorial(n + 1)
    return x_grid, an_error

#find_y(arr, z)
N = 400
x_grid, error1 = interpolation(np.array([1 + k / N for k in range(N + 1)]), (lambda t: np.log(t)))
x_grid, error2 = max_error(np.array([1 + k / N for k in range(N + 1)]), lambda t: -5040/(t**8))

plt.plot(x_grid, error2)
plt.plot(x_grid, error1)
plt.show()

