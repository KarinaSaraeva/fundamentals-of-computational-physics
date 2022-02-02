import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import animation
from matplotlib import cm
matplotlib.use('Qt5Agg')

def banded_matrix(h, tau, n):
    matrix = np.zeros((n, n))

    for i in range(n - 1):
        matrix[i, i] = 1 + tau/(h**2)
        matrix[i, i + 1] = -(0.5)*tau/(h**2)
        matrix[i + 1, i] = -(0.5)*tau/(h**2)

    matrix[n - 1, n - 1] = 1 + tau/(h**2)
    return matrix


def gauss_for_banded(matrix, d):
    buffer = np.copy(matrix)
    for i in range(len(buffer[0]) - 1):
        temp = buffer[i + 1, i] / buffer[i, i]
        buffer[i + 1, :] = buffer[i + 1, :] - buffer[i, :] * temp
        d[i + 1] = d[i + 1] - d[i] * temp

    return buffer, d


def x_for_gauss(matrix, d):
    n = len(d)
    x = np.zeros(n)

    x[n - 1] = d[n - 1]/matrix[n - 1][n - 1]
    for i in range(n - 1):
        x[n - 2 - i] = (1/matrix[n - 2 - i][n - 2 - i])*(d[n - 2 - i] - matrix[n - 2 - i][n - i - 1]*(x[n - 1 - i]))

    return x


def sweep_diffusion(func, a, b, T_max, N_x, N_t, y_a, y_b):

    x_grid = np.linspace(a, b, N_x)
    t_grid = np.linspace(0, T_max, N_t)
    h = x_grid[1]-x_grid[0]
    tau = t_grid[1]-t_grid[0]
    matrix = banded_matrix(h, tau, N_x - 2)

    u = np.zeros((N_t, N_x))
    u[0, :] = func(x_grid)
    u[0, 0] = y_a
    u[0, N_x - 1] = y_b
    plt.plot(x_grid, u[0, :])
    right_part = np.zeros(N_x - 2)


    for n in range(len(t_grid) - 1):
        for i in range(0, len(right_part)):
            right_part[i] = u[n, i + 1] + (tau / 2) * ((u[n, i + 2] - 2 * u[n, i + 1] + u[n, i]) / (h**2))

        triagonal, newright_part = gauss_for_banded(matrix, right_part)

        u[n + 1, 1:(N_x - 1)] = x_for_gauss(triagonal, newright_part)

        u[n + 1, 0] = y_a
        u[n + 1, N_x - 1] = y_b


    return t_grid, x_grid, u


def draw_animation(t_grid, x_grid, u):

    fig = plt.figure()
    ax = plt.axes()
    line, = ax.plot([], [], lw=2)

    def init():
        line.set_data([], [])
        ax.set_xlim(0, x_grid[-1])
        ax.set_ylim(0, np.amax(u[0, :]))
        return line,

    x = x_grid

    def animate(i):
        y = u[int(i), :]
        line.set_data(x, y)
        return line,

    anim = animation.FuncAnimation(fig, animate, frames=np.linspace(0, len(t_grid), len(t_grid)), init_func=init, interval=20, blit=True)
    plt.show()

    return 0

L = 1
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

t_grid, x_grid, arr_u = sweep_diffusion(lambda x: x*((1 - x/L)**2), 0, L, 0.2, 100, 300, 0, 0)


X, Y = np.meshgrid(x_grid, t_grid)
surf = ax.plot_surface(X, Y, arr_u, cmap=cm.coolwarm, linewidth=0, antialiased=False)
fig.colorbar(surf, shrink=0.5, aspect=5)
plt.show()

draw_animation(t_grid, x_grid, arr_u)