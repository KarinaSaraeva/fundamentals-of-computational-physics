import numpy as np
import matplotlib.pyplot as plt
a = 100
b = 2
c = 2
d = 100


def y_func(vec, t = None): # x and y
    return np.array([a*vec[0] - b*vec[0]*vec[1], c*vec[0]*vec[1] - d*vec[1]])


def runge2(func, start_t, stop_t, N_t):
    t_grid = np.linspace(start_t, stop_t, N_t)

    x_arr = np.zeros(N_t)
    x_arr[0] = 5

    y_arr = np.zeros(N_t)
    y_arr[0] = 10

    vec = np.vstack((x_arr, y_arr))
    h = (t_grid[1] - t_grid[0])
    a = 1/4
    for i in range(len(t_grid) - 1):
        k1 = func(vec[:, i], t_grid[i])
        k2 = func(vec[:, i] + k1*h/(2*a), t_grid[i] + h/(2*a))
        vec[:, i+1] = (vec[:, i] + h * ((1 - a) * k1 + a * k2))

    return vec


N = 1000
start, stop = 0, 10
grid = runge2(y_func, start, stop, N)
plt.plot(np.linspace(start, stop, N), grid[1,:])
plt.plot(np.linspace(start, stop, N), grid[0,:])
plt.show()
plt.plot(grid[0,:], grid[1,:])
plt.show()