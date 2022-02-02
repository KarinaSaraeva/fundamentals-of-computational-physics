import numpy as np
import matplotlib.pyplot as plt


def y_func(vec, t = None): # x and y
    return np.array([998*vec[0] + 1998*vec[1], -999*vec[0] - 1999*vec[1]])



def euler(func, start_t, stop_t, x_start, y_start, N_t):
    t_grid = np.linspace(start_t, stop_t, N_t)

    x_arr = np.zeros(N_t)
    x_arr[0] = x_start

    y_arr = np.zeros(N_t)
    y_arr[0] = y_start

    vec = np.vstack((x_arr, y_arr))
    h = (t_grid[1] - t_grid[0])
    for i in range(len(t_grid) - 1):
        k1 = func(vec[:, i], t_grid[i])
        vec[:, i+1] = vec[:, i] + h * k1

    return vec


def implicit_euler(func, J, start_t, stop_t, x_start, y_start, N_t):
    t_grid = np.linspace(start_t, stop_t, N_t)

    x_arr = np.zeros(N_t)
    x_arr[0] = x_start

    y_arr = np.zeros(N_t)
    y_arr[0] = y_start

    vec = np.vstack((x_arr, y_arr))

    h = t_grid[1]-t_grid[0]
    eig_val = np.linalg.eigvals(h * J)

    if np.all(abs(eig_val) < 1):
        for i in range(len(t_grid)-1):
            temp = h*func(vec[:, i], t_grid[i])
            for j in range(10):
                temp = h*np.dot(J, temp) + h*func(vec[:, i], t_grid[i])
            vec[:, i + 1] = vec[:, i] + temp
    else:
        print("unstable, increase N_t")
        print(abs(eig_val) < 1)
    return vec


def true_func(t, x_0, y_0):
    a = x_0 + y_0
    b = -x_0 - 2 * y_0
    return (2*a*np.exp(-t)+b*np.exp(-1000*t)), (-a*np.exp(-t) - b*np.exp(-1000*t))

N = 50000
start, stop = 0, 5
x_0 = 5
y_0 = 10

t_grid = np.linspace(start, stop, N)
grid1 = euler(y_func, start, stop, x_0, y_0, N)
plt.plot(t_grid, grid1[1, :])
plt.plot(t_grid, grid1[0, :])
plt.title('euler')
plt.show()

grid_true = true_func(t_grid, x_0, y_0)
plt.plot(t_grid, grid1[0, :] - grid_true[0])
plt.plot(t_grid, grid1[1, :] - grid_true[1])
plt.ylim([-0.002, 0.002])
plt.title('euler error')
plt.show()

grid2 = implicit_euler(y_func, np.array([[998, 1998], [-999, -1999]]), start, stop, x_0, y_0, N)
plt.plot(t_grid, grid2[1, :])
plt.plot(t_grid, grid2[0, :])
plt.title('implicit euler')
plt.show()


plt.plot(t_grid, grid2[0, :] - grid_true[0])
plt.plot(t_grid, grid2[1, :] - grid_true[1])
plt.ylim([-0.002, 0.002])
plt.title('implicit euler error')
plt.show()