import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from itertools import product
from warnings import warn
from sklearn.datasets import make_spd_matrix


def LinearCG(A, b, x0, tol=1e-5):
    xk = x0
    rk = np.dot(A, xk) - b
    pk = -rk
    rk_norm = np.linalg.norm(rk)

    num_iter = 0
    curve_x = [xk]
    while rk_norm > tol:
        apk = np.dot(A, pk)
        rkrk = np.dot(rk, rk)

        alpha = rkrk / np.dot(pk, apk)
        xk = xk + alpha * pk
        rk = rk + alpha * apk
        beta = np.dot(rk, rk) / rkrk
        pk = -rk + beta * pk

        num_iter += 1
        curve_x.append(xk)
        rk_norm = np.linalg.norm(rk)
        print('Iteration: {} \t x = {} \t residual = {:.4f}'.
              format(num_iter, xk, rk_norm))

    print('\nSolution: \t x = {}'.format(xk))

    return np.array(curve_x)


def create_mesh(f):
    x = np.arange(-5, 5, 0.025)
    y = np.arange(-5, 5, 0.025)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros(X.shape)
    mesh_size = range(len(X))
    for i, j in product(mesh_size, mesh_size):
        x_coor = X[i][j]
        y_coor = Y[i][j]
        Z[i][j] = f(np.array([x_coor, y_coor]))
    return X, Y, Z


def plot_contour(ax, X, Y, Z):
    ax.set(
        title='Path During Optimization Process',
        xlabel='x1',
        ylabel='x2'
    )
    CS = ax.contour(X, Y, Z)
    ax.clabel(CS, fontsize='smaller', fmt='%1.2f')
    ax.axis('square')
    return ax


def f(x):
    Ax = np.dot(A, x)
    xAx = np.dot(x, Ax)
    bx = np.dot(b, x)
    return 0.5 * xAx - bx


np.random.seed(0)
A = make_spd_matrix(2, random_state=0)
# A = make_spd_matrix(2)
x_star = np.random.random(2)
b = np.dot(A, x_star)
x0 = np.array([-3, -4])
xs = LinearCG(A, b, x0)

print('A\n', A, '\n')
print('b\n', b, '\n')
print('The solution x* should be\n', x_star)
print('A\n', A, '\n')
print('b\n', b, '\n')
print('The solution x* should be\n', x_star)
fig, ax = plt.subplots(figsize=(6, 6))
X, Y, Z = create_mesh(f)
ax = plot_contour(ax, X, Y, Z)
ax.plot(xs[:, 0], xs[:, 1], linestyle='--', marker='o', color='orange')
ax.plot(xs[-1, 0], xs[-1, 1], 'ro')
plt.show()

# np.random.seed(0)
A = make_spd_matrix(3, random_state=0)
x_star = np.random.random(3)
b = np.dot(A, x_star)

print('A\n', A, '\n')
print('b\n', b, '\n')
print('The solution x* should be\n', x_star)

x0 = np.array([3, 1, -7])
xs = LinearCG(A, b, x0)

# fig, ax = plt.subplots(figsize=(6, 6))
# X, Y, Z = create_mesh(f)
# ax = plot_contour(ax, X, Y, Z)
# ax.plot(xs[:, 0], xs[:, 1], linestyle='--', marker='o', color='orange')
# ax.plot(xs[-1, 0], xs[-1, 1], 'ro')
# plt.show()