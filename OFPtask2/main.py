import numpy as np
import matplotlib.pyplot as plt

U = 100
a_0 = 1
accuracy = 10 ** (-10)

y = lambda e: 1 / (np.tan(np.sqrt(2 * (a_0 ** 2) * (U + e)))) - np.sqrt(-U / e - 1)
diff_y = lambda e: -a_0/((np.sin(np.sqrt(2*a_0**2*(U+e)))**2)*np.sqrt((U+e)*2))-U/(2*e**2*np.sqrt(-U/e-1))
x = np.linspace(-0.999 * U, -0.001 * U, 1000)


plt.plot(x, y(x))
plt.show()

def find_area(inter, func):
    for i in range(len(inter) - 1):
        if (func(inter[i]) * func(inter[i + 1])) < 0:
            print('a =', inter[i], 'b =', inter[i + 1])
            return inter[i], inter[i + 1]
    return "error"


def halfing(a, b, func, delta):
    while (b - a) > delta:
        if (func((a + b) / 2) * func(a)) < 0:
            b = (a + b) / 2
        else:
            a = (a + b) / 2
    print("by halfing method the ground state is ", b)
    return b


def iteration(a, b, func, delta):

    lamb = 1/((func(b) - func(a))/(b - a))
    x_arr = [a, a - lamb * func(a), (a - lamb * func(a)) - lamb * func(a - lamb * func(a))]
    while (((x_arr[2] - x_arr[1]) ** 2) / abs(2 * x_arr[1] - x_arr[0] - x_arr[2])) > delta:
        x_arr[0] = x_arr[1]  # x_n_2
        x_arr[1] = x_arr[2]  # x_n_1
        x_arr[2] = x_arr[1] - lamb * func(x_arr[1]) #x_n
    print("by iteration method the ground state is ", x_arr[2])
    return x[2]

def Newtown(a, b, func, delta):

    lamb = 1/diff_y(a)
    x_arr = [a, a - lamb * func(a), (a - lamb * func(a)) - lamb * func(a - lamb * func(a))]
    while (((x_arr[2] - x_arr[1]) ** 2) / abs(2 * x_arr[1] - x_arr[0] - x_arr[2])) > delta:
        x_arr[0] = x_arr[1]  # x_n_2
        x_arr[1] = x_arr[2]  # x_n_1
        lamb = 1 / diff_y(x_arr[1])
        x_arr[2] = x_arr[1] - lamb * func(x_arr[1]) #x_n
    print("by Newtown method the ground state is ", x_arr[2])
    return x[2]

c, d = find_area(x, y)
plt.plot(x, y(x))
plt.xlim([c, d])
plt.ylim([y(c) / 10, y(d)])
plt.axhline(color='black')
e = halfing(c, d, y, accuracy)
e1 = iteration(c, d, y, accuracy)
e2 = Newtown(c, d, y, accuracy)
plt.scatter(e, 0, c='r')
#plt.scatter(e1, 0, c='r')

plt.show()
