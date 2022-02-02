import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
def U(x):
    return (1/2)*x**2

def banded_matrix(a, b, c, h, n):
    matrix = np.zeros((n, n))
    for i in range(n - 1):
        matrix[i][i] = -b / (h ** 2)
        matrix[i][i + 1] = a / (h ** 2)
        matrix[i + 1][i] = c / (h ** 2)

    matrix[0][0] = 1
    matrix[0][1] = 0
    matrix[n - 1][n - 2] = 0
    matrix[n - 1][n - 1] = 1
    return matrix

def U_matrix(U, n, x_grid):
    matrix = np.zeros((n, n))
    for i in range(n):
        matrix[i][i] = U(x_grid[i])
    matrix[0][0] = 1
    matrix[0][1] = 0
    matrix[n - 1][n - 2] = 0
    matrix[n - 1][n - 1] = 1
    return matrix


def find_eigval(matrix):

    initial = np.array([1.0 for k in range(len(matrix[0, :]))])
    curr = initial
    current_eig1 = 0
    current_eig2 = 10

    while abs(current_eig1 - current_eig2) > 0.002:
        prev = curr
        curr = np.dot(matrix, prev)
        current_eig1 = curr[1] / prev[1]
        current_eig2 = curr[0] / prev[0]

    eig = current_eig1
    return eig, curr


def schrodinger_eq(a, b, N_x, U):

    x_grid = np.linspace(a, b, N_x)
    h = x_grid[1] - x_grid[0]
    d2x_matrix = banded_matrix(1, 2, 1, h, N_x)
    u_matrix = U_matrix(U, N_x, x_grid)
    matrixH = -(0.5)*d2x_matrix + u_matrix
    temp, wave_fun = find_eigval(np.linalg.inv(matrixH))
    Energy_min = 1/temp

    print(Energy_min)

    plt.plot(x_grid, wave_fun)
    plt.show()

    return 0

# A = np.array([[10, 80, 30], [40, 50, 60], [70, 80, 90]])
# print(A)
# print(find_eigval(A))

schrodinger_eq(-20, 20, 4000, U)

