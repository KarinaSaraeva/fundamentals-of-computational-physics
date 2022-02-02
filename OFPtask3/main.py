import matplotlib.pyplot as plt
import numpy as np

y = lambda x: 1 / (1 + x ** 2)
y1 = lambda x: x**(1/3)*np.exp(np.sin(x))

def trapezoid(N, func, a, b):
    x = np.linspace(a, b, N + 1)
    I = 0
    for i in range(N):
        I += ((func(x[i]) + func(x[i + 1])) / 2) * (x[i + 1] - x[i])

    print("Integral calculated with trapezoid method is", I)
    return I

def Simpson(N, func, a, b):
    x = np.linspace(a, b, N + 1)
    I = 0
    for i in range(N):
        I += (func(x[i]) + 4*func((x[i] + x[i + 1])/2) + func(x[i + 1])) * ((x[i + 1] - x[i])/6)

    print("Integral calculated with Simpson method is", I)
    return I

def accuracy_check(N_1, N_2, func, a, b, method_name, title, I_true):
    N_curr = N_1
    R_arr = []
    while N_curr < N_2:
        R_arr.append(I_true - method_name(N_curr, func, a, b))
        N_curr += 1
    plt.plot(np.linspace(N_1, N_2, N_2 - N_1), R_arr)
    plt.title(title)
    plt.show()
    return R_arr



#accuracy_check(5, 100, y, -1, 1, trapezoid, 'trapezoid R', np.pi/2)
N_from = 4
N_to = 100
Arr_1 = accuracy_check(N_from, N_to, y, -1, 1, Simpson, 'Simpson R', np.pi/2)
Arr_2 = accuracy_check(N_from, N_to, y, -1, 1, trapezoid, 'trapezoid R', np.pi/2)
fig, ax = plt.subplots(2, 1)
line_space = np.linspace(N_from, N_to, N_to - N_from)
ax[0].plot(line_space, Arr_1, label = 'Simpson R')

ax[0].plot(line_space, Arr_2, label = 'trapezoid R')
ax[1].plot(line_space, Arr_1, label = 'Simpson R')
ax[1].plot(line_space, Arr_2, label = 'trapezoid R')

ax[1].set_yscale('log')

ax[0].legend()
ax[1].legend()
plt.show()